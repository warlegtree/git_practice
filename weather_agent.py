import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from weather_tool import get_weather

load_dotenv()  # 加载 .env 中的 OPENAI_API_KEY / OPENAI_BASE_URL

client = OpenAI()  # 自动读取环境变量

# 从 JSON 文件加载工具定义
tools_schema = json.loads(
    (Path(__file__).parent / "tools_schema.json").read_text(encoding="utf-8")
)

# 工具名 → 函数 的分发表，新增工具时在这里注册即可
TOOL_DISPATCH = {
    "get_weather": get_weather,
}

MAX_ROUNDS = 10  # 最大工具调用轮数，防止死循环


def execute_tool(func_name: str, func_args: dict) -> dict:
    """按名称分发并执行工具，异常时返回错误信息而不是崩溃"""
    func = TOOL_DISPATCH.get(func_name)
    if func is None:
        return {"error": f"未知工具: {func_name}"}
    try:
        return func(**func_args)
    except TypeError as e:
        return {"error": f"工具参数错误: {e}"}


def run_agent(user_message: str) -> str:
    """运行 Agent：感知→决策→行动→观察"""
    messages = [
        {"role": "system", "content": "你是一个天气助手，帮用户查询天气。用自然语言回答。"},
        {"role": "user", "content": user_message},
    ]

    # === Agent 循环 ===
    for _ in range(MAX_ROUNDS):
        # 1. 感知 + 决策：AI 决定是否调用工具
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        # 2. 判断：AI 没调用工具 → 任务完成，返回结果
        if not msg.tool_calls:
            return msg.content or ""

        # 3. 行动：AI 要调工具 → 执行工具
        messages.append(msg)  # 先把 AI 的消息加入历史

        for tool_call in msg.tool_calls:
            # 解析工具名和参数
            func_name = tool_call.function.name
            try:
                func_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                func_args = {}

            print(f"🔧 调用工具: {func_name}({func_args})")

            # 执行工具
            result = execute_tool(func_name, func_args)

            # 4. 观察：把工具结果返回给 AI
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        # 循环回去 → AI 看到工具结果，决定下一步

    return "抱歉，处理超时：已超过最大工具调用轮数。"


if __name__ == "__main__":
    # === 测试 ===
    print(run_agent("武汉明天天气怎么样？能出去玩吗？"))
