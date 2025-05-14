from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-xxx")

completion = client.chat.completions.create(
  model="Qwen3-4b",
  messages=[
    {"role": "system", "content": "你是一个有逻辑的助手"},
    {"role": "user", "content": "1+1是多少？"}
  ],
  stream=True
)

for chunk in completion:
    print(chunk.choices[0].delta)
