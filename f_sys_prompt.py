from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('templates'))
template = env.get_template('agent_system.j2')
final_system_prompt = template.render(
    agent_name="天气助手",
    role_description="专业的气象服务 Agent",
    tools=[
        {"name": "get_weather", "description": "查询指定城市天气",
         "parameters": [
            {"name": "city", "type": "string", "description": "城市名称"},
            {"name": "date", "type": "string", "description": "日期，默认今天"}
         ]}
    ],
    constraints=[
        "只能回答天气相关问题",
        "如果查询不到数据，如实告知用户",
        "输出必须是合法 JSON"
    ],
    examples=[
        {"user": "北京今天天气怎么样？",
         "assistant": '{"action":"call_tool","tool":"get_weather","args":{"city":"北京"}}'}
    ],
    output_schema='{"action": "call_tool|respond", "tool": "工具名", "args": {}, "answer": ""}'
)

print(final_system_prompt)