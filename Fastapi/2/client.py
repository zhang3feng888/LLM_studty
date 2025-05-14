import requests

url = "http://0.0.0.0:6665/chat/"
query = {"text": "你好，请做一段自我介绍。"}

response = requests.post(url, json=query)

if response.status_code == 200:
    result = response.json()
    print("BOT:", result["result"])
else:
    print("Error:", response.status_code, response.text)
