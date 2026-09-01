import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, RateLimitError, APIError

from weather_tool import get_weather

load_dotenv()  # 加载 .env 中的 OPENAI_API_KEY / OPENAI_BASE_URL

client = OpenAI()  # 自动读取环境变量

MAX_RETRIES = 3  # API 调用最大重试次数

# 工具名 → 函数 的分发表，新增工具时在这里注册即可
TOOL_DISPATCH = {
    "get_weather": get_weather,
}

# 从 JSON 文件加载工具定义，只保留本 Agent 支持的工具
# （schema 文件中还定义了 plan_trip 等其他工具，纯天气助手不需要）
_all_tools = json.loads(
    (Path(__file__).parent / "tools_schema.json").read_text(encoding="utf-8")
)
tools_schema = [t for t in _all_tools if t["function"]["name"] in TOOL_DISPATCH]

MAX_ROUNDS = 10  # 单次提问内最大工具调用轮数，防止死循环

EXIT_WORDS = {"quit", "exit", "bye", "退出"}


def execute_tool(func_name: str, func_args: dict) -> dict:
    """按名称分发并执行工具，异常时返回错误信息而不是崩溃"""
    func = TOOL_DISPATCH.get(func_name)
    if func is None:
        return {"error": f"未知工具: {func_name}"}
    try:
        return func(**func_args)
    except TypeError as e:
        return {"error": f"工具参数错误: {e}"}
    except Exception as e:
        return {"error": f"工具执行异常: {e}"}


def call_llm_with_retry(messages: list, tools: list):
    """带重试的 LLM 调用，处理超时和限流"""
    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=messages,
                tools=tools,
                timeout=30,
            ), None
        except APITimeoutError:
            if attempt < MAX_RETRIES - 1:
                print(f"⏱️ API 超时，第 {attempt + 1} 次重试...")
                time.sleep(2 ** attempt)
            else:
                return None, "API 请求超时，请稍后重试"
        except RateLimitError:
            wait = min(2 ** (attempt + 2), 60)
            print(f"🚫 服务限流，等待 {wait}s 后重试...")
            time.sleep(wait)
            if attempt == MAX_RETRIES - 1:
                return None, "服务繁忙，请稍后重试"
        except APIError as e:
            return None, f"API 错误: {e.message}"
        except Exception as e:
            return None, f"未知错误: {e}"
    return None, "请求失败"


def run_agent_loop():
    """多轮对话 Agent：messages 在对话循环外持久化，每轮提问内部跑 感知→决策→行动→观察"""
    messages = [
        {"role": "system", "content": "你是一个天气助手，帮用户查询天气。用自然语言回答。"}
    ]

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

        # === Agent 循环：一次提问内可能有多轮工具调用 ===
        for _ in range(MAX_ROUNDS):
            # 1. 感知 + 决策：AI 决定是否调用工具（带重试）
            response, error = call_llm_with_retry(messages, tools_schema)
            if error:
                print(f"助手：{error}")
                break

            msg = response.choices[0].message

            # 2. 判断：AI 没调用工具 → 本轮任务完成
            if not msg.tool_calls:
                reply = msg.content or ""
                print(f"助手：{reply}")
                messages.append({"role": "assistant", "content": reply})
                break

            # 3. 行动：AI 要调工具 → 执行工具
            messages.append(msg)

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                # JSON 解析失败处理
                try:
                    func_args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON 解析失败: {e}")
                    result = {"error": f"参数解析失败: {e}"}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    continue

                print(f"🔧 调用工具: {func_name}({func_args})")
                result = execute_tool(func_name, func_args)

                # 4. 观察：把工具结果返回给 AI
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
        else:
            print("助手：抱歉，处理超时：已超过最大工具调用轮数。")


if __name__ == "__main__":
    # === 测试 ===
    run_agent_loop()
