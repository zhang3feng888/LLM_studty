# 连续对话

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-xxx",  # 随便写，FastAPI那边不会校验
)

# 初始化对话历史
conversation_history = [
    {"role": "system", "content": "你是一个非常有用的智能助手。"}
]

print("欢迎进入 Qwen 对话系统，输入内容开始对话，输入 'exit' 退出。\n")

while True:
    # 获取用户输入
    user_input = input("你：")
    if user_input.lower().strip() in ["exit", "quit"]:
        print("结束对话。")
        break

    # 添加用户消息到对话历史
    conversation_history.append({"role": "user", "content": user_input})

    # 调用模型生成回答
    response = client.chat.completions.create(
        model="Qwen3-4b",
        messages=conversation_history
    )
    assistant_reply = response.choices[0].message.content.strip()

    # 打印助手回复并加入对话历史
    print(f"助手：{assistant_reply}")
    conversation_history.append({"role": "assistant", "content": assistant_reply})
