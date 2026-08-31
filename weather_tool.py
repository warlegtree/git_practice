from datetime import datetime, timedelta


def get_weather(city: str, date: str = "today") -> dict:
    """
    查询指定城市的天气

    Args:
        city: 城市名称，如 "北京"、"上海"
        date: 日期，"today"、"tomorrow" 或 "YYYY-MM-DD" 格式

    Returns:
        包含天气信息的字典
    """
    # 模拟天气数据（实际项目调用天气 API）
    weather_data = {
        "北京": {"today": ("晴", 25), "tomorrow": ("多云", 23)},
        "上海": {"today": ("小雨", 28), "tomorrow": ("小雨", 27)},
        "广州": {"today": ("雷阵雨", 31), "tomorrow": ("晴", 33)},
    }

    if city not in weather_data:
        return {"error": f"暂不支持查询 {city} 的天气，目前仅支持：{'、'.join(weather_data)}"}

    # 处理日期：统一映射到 today / tomorrow，并解析出实际日期
    today = datetime.now().date()
    if date == "today":
        target = today
    elif date == "tomorrow":
        target = today + timedelta(days=1)
    else:
        try:
            target = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            return {"error": f"无法识别的日期: {date}，请使用 today、tomorrow 或 YYYY-MM-DD 格式"}

    delta = (target - today).days
    if delta == 0:
        date_key = "today"
    elif delta == 1:
        date_key = "tomorrow"
    else:
        return {"error": f"仅支持查询今天和明天的天气，{target.isoformat()} 超出范围"}

    weather, temp = weather_data[city][date_key]
    return {
        "city": city,
        "date": target.isoformat(),
        "weather": weather,
        "temperature": f"{temp}°C",
    }


if __name__ == "__main__":
    # 测试工具
    print(get_weather("武汉", "明天"))
    # {'city': '北京', 'date': '2026-08-21', 'weather': '多云', 'temperature': '23°C'}
