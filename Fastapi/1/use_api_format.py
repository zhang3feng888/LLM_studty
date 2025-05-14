from openai import OpenAI

if __name__ == "__main__":
    # 使用本地 API 服务
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")  # 这里可以填你自己的API_KEY

    response = client.chat.completions.create(
        model="Qwen3-4B",
        messages=[{"role": "user", "content": "你好"}],
        stream=False
    )

    # 遍历流式响应并打印
    for chunk in response:
        if 'content' in chunk.choices[0].delta:
            print(chunk.choices[0].delta['content'], end="", flush=True)
