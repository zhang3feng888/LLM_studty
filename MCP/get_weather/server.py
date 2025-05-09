import json
 
import httpx
from mcp.server import FastMCP
 
mcp = FastMCP("WeatherServer") # 名字可以随便起吗？
 
OPENWEATHER_API_KEY = "b2ca735bb0e860cc3220c01e1b30e9cf"
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
USER_AGENT = "weather-app/1.0"
 
async def get_weather(city):
    """
    从OpenWeather API 获取天气信息
    :param city: 城市名称（需要使用英文，如 beijing）
    :return: 天气数据字典；若发生错误，返回包含error信息的字典
    """
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric", # 度量衡，摄氏度
        "lang": "zh_cn", # 返回中文描述
    }
    headers = {"User-Agent": USER_AGENT}
 
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(OPENWEATHER_BASE_URL, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP请求错误：{e}"}
        except Exception as e:
            return {"error": f"发生错误：{e}"}
 
def format_weather_data(data):
    """
    格式化天气数据
    :param data: 天气数据字典
    :return: 格式化后的字符串；若发生错误，返回包含error信息的字符串
    """
 
    #  如果传入的是字符串，则先转换成字典
    if isinstance(data, str):
        data = json.loads(data)
 
    if "error" in data:
        return data["error"]
    
    # 这些字段是根据 OpenWeather API 官方文档 所规定的 JSON 响应结构写死的。
    # 如果你换成别的 API，结构可能完全不同，代码也就要相应修改。
    weather = data["weather"][0]["description"]
    temperature = data["main"]["temp"]
    city = data["name"]
    country = data["sys"]["country"]
    humidity = data["main"]["humidity"]
    wind = data["wind"]["speed"]
 
    return f"城市：{city}, {country}\n天气：{weather}\n温度：{temperature}°C\n湿度：{humidity}%\n风速：{wind}m/s"
 
 
@mcp.tool()
async def get_weather_tool(city: str):
    """
    获取城市的天气信息
    :param city: 城市名称（需要试用英文，如 beijing）
    :return: 天气数据字典；若发生错误，返回包含error信息的字典
    """
    weather_data = await get_weather(city)
    return format_weather_data(weather_data)
 
 
if __name__ == "__main__":
    mcp.run(transport="stdio")