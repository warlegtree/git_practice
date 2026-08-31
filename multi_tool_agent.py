import json
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from trip_plan_tool import plan_trip
from weather_tool import get_weather

load_dotenv()  # 加载 .env 中的 OPENAI_API_KEY / OPENAI_BASE_URL

client = OpenAI()  # 自动读取环境变量

MODEL = "deepseek-v4-flash"

# === 计费规则（每百万 tokens，人民币元）===
# deepseek-v4-flash 官方价格；其他模型拿到价格表后在此补充
PRICING = {
    "deepseek-v4-flash": {
        "peak":     {"cache_hit": 0.10, "cache_miss": 3.0, "output": 9.0},
        "off_peak": {"cache_hit": 0.05, "cache_miss": 1.5, "output": 4.5},
    },
}

# DeepSeek 优惠时段：UTC 16:30 – 次日 00:30（北京时间 00:30 – 08:30）
OFF_PEAK_START_MIN = 16 * 60 + 30  # UTC 16:30
OFF_PEAK_END_MIN = 30              # UTC 00:30


def is_off_peak(now=None) -> bool:
    """判断当前是否处于 off-peak 优惠时段（跨零点：16:30–24:00 或 00:00–00:30）"""
    now = now or datetime.now(timezone.utc)
    minutes = now.hour * 60 + now.minute
    return minutes >= OFF_PEAK_START_MIN or minutes < OFF_PEAK_END_MIN


class CallTracker:
    """记录 Agent 运行中的调用链、token 消耗、延迟与预估成本"""

    def __init__(self, model: str):
        self.model = model
        self.off_peak = is_off_peak()
        self.llm_calls = 0
        self.tool_calls = []  # 工具调用链，按调用顺序记录工具名
        self.prompt_tokens = 0
        self.cache_hit_tokens = 0
        self.cache_miss_tokens = 0
        self.completion_tokens = 0
        self.llm_latency = 0.0
        self.cost = 0.0

    def record_llm_call(self, round_no: int, usage, latency: float) -> None:
        """记录一轮 LLM 调用的 token 消耗、延迟与成本，并打印本轮明细"""
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        # DeepSeek 的 usage 会细分缓存命中/未命中；没有该字段时全部按未命中计
        hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        miss = getattr(usage, "prompt_cache_miss_tokens", None)
        miss = miss if miss is not None else prompt - hit

        price = PRICING[self.model]["off_peak" if self.off_peak else "peak"]
        cost = (hit * price["cache_hit"]
                + miss * price["cache_miss"]
                + completion * price["output"]) / 1_000_000

        self.llm_calls += 1
        self.llm_latency += latency
        self.prompt_tokens += prompt
        self.cache_hit_tokens += hit
        self.cache_miss_tokens += miss
        self.completion_tokens += completion
        self.cost += cost

        print(f"📊 第 {round_no} 轮 LLM 调用 | 延迟 {latency:.2f}s | "
              f"tokens: 输入 {prompt} (缓存命中 {hit} / 未命中 {miss}) + 输出 {completion} | "
              f"本轮成本 ¥{cost:.6f}")

    def merge(self, other: "CallTracker") -> None:
        """把另一段统计累加进来（用于会话级汇总）"""
        self.llm_calls += other.llm_calls
        self.tool_calls.extend(other.tool_calls)
        self.prompt_tokens += other.prompt_tokens
        self.cache_hit_tokens += other.cache_hit_tokens
        self.cache_miss_tokens += other.cache_miss_tokens
        self.completion_tokens += other.completion_tokens
        self.llm_latency += other.llm_latency
        self.cost += other.cost

    def print_summary(self, title: str = "本次调用统计") -> None:
        """打印汇总统计"""
        period = "off-peak 优惠时段" if self.off_peak else "peak 标准时段"
        chain = " → ".join(self.tool_calls) if self.tool_calls else "（无工具调用）"
        total_tokens = self.prompt_tokens + self.completion_tokens
        print(f"\n📈 ===== {title}（{period}）=====")
        print(f"   工具调用链: {chain}")
        print(f"   LLM 调用: {self.llm_calls} 次 | 工具调用: {len(self.tool_calls)} 次 | "
              f"LLM 总延迟: {self.llm_latency:.2f}s")
        print(f"   Tokens: 输入 {self.prompt_tokens} "
              f"(缓存命中 {self.cache_hit_tokens} / 未命中 {self.cache_miss_tokens}) "
              f"+ 输出 {self.completion_tokens} = {total_tokens}")
        print(f"   预估成本: ¥{self.cost:.6f}")


# 从 JSON 文件加载工具定义
tools_schema = json.loads(
    (Path(__file__).parent / "tools_schema.json").read_text(encoding="utf-8")
)

# 工具名 → 函数 的分发表，新增工具时在这里注册即可
TOOL_DISPATCH = {
    "get_weather": get_weather,
    "plan_trip": plan_trip,
}

MAX_ROUNDS = 10  # 单次提问内最大工具调用轮数，防止死循环
MAX_RETRIES = 2  # 工具执行异常时的重试次数

EXIT_WORDS = {"quit", "exit", "bye", "退出"}


def execute_tool_with_retry(func_name: str, func_args: dict) -> dict:
    """
    按名称分发并执行工具，异常时自动重试。

    注意：TypeError（参数错误）是确定性的，重试无意义，直接返回错误；
    其他异常（如网络抖动）才值得重试。
    """
    func = TOOL_DISPATCH.get(func_name)
    if func is None:
        return {"error": f"未知工具: {func_name}"}

    for attempt in range(MAX_RETRIES + 1):
        try:
            return func(**func_args)
        except TypeError as e:
            return {"error": f"工具参数错误: {e}"}
        except Exception as e:
            if attempt >= MAX_RETRIES:
                return {"error": f"工具执行失败（已重试 {MAX_RETRIES} 次）: {e}"}
            time.sleep(0.5 * (attempt + 1))  # 简单退避


def chat_once(messages: list, tracker: CallTracker) -> str:
    """单轮提问的 Agent 循环：感知→决策→行动→观察，返回助手回复"""
    for round_no in range(1, MAX_ROUNDS + 1):
        # 1. 感知 + 决策：AI 决定是否调用工具
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools_schema,
            )
        except Exception as e:
            return f"AI 服务暂时不可用: {e}"
        latency = time.perf_counter() - start

        tracker.record_llm_call(round_no, response.usage, latency)

        msg = response.choices[0].message

        # 2. 判断：AI 没调用工具 → 本轮任务完成
        if not msg.tool_calls:
            reply = msg.content or ""
            # 关键：把 AI 的回复也记入历史，否则下一轮对话丢失上下文
            messages.append({"role": "assistant", "content": reply})
            return reply

        # 3. 行动：AI 要调工具 → 执行工具
        messages.append(msg)  # 先把 AI 的消息加入历史

        for tool_call in msg.tool_calls:
            # 解析工具名和参数
            func_name = tool_call.function.name
            try:
                func_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                func_args = {}

            print(f"🔧 调用: {func_name}({func_args})")

            # 使用带重试的执行，并记录工具耗时
            tool_start = time.perf_counter()
            result = execute_tool_with_retry(func_name, func_args)
            tool_latency = time.perf_counter() - tool_start
            tracker.tool_calls.append(func_name)

            print(f"📦 结果: {result} (耗时 {tool_latency:.3f}s)")

            # 4. 观察：把工具结果返回给 AI
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        # 循环回去 → AI 看到工具结果，决定下一步

    return "抱歉，处理超时：已超过最大工具调用轮数。"


def run_agent_loop():
    """多轮对话入口：messages 跨轮持久化；输入 exit/quit/bye/退出 结束对话并打印总账"""
    messages = [
        {
            "role": "system",
            "content": "你是一个出行助手，可以查询天气和规划行程。\n"
                       "如果用户问出行建议，先查天气再规划行程。\n"
                       "如果工具返回错误，根据错误信息调整策略。",
        }
    ]
    session = CallTracker(MODEL)  # 累计整个会话的消耗

    # === 多轮对话循环 ===
    while True:
        try:
            user_input = input("用户：").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+C / Ctrl+D 也能优雅退出
            print("\nbye")
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_WORDS:
            print("bye")
            break

        messages.append({"role": "user", "content": user_input})

        turn = CallTracker(MODEL)  # 每轮提问单独统计
        reply = chat_once(messages, turn)
        print(f"助手：{reply}")
        turn.print_summary("本轮对话")
        session.merge(turn)

    # 会话结束，打印总消耗
    if session.llm_calls:
        session.print_summary("会话总计")


if __name__ == "__main__":
    run_agent_loop()
