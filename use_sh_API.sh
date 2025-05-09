# .sh的方式使用API

curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "DeepSeek-R1-Distill-Qwen-7B",
        "messages": [
            {"role": "system", "content": "你是一个职业技术学习专家。"},
            {"role": "user", "content": "学挖掘机技术哪家强？"}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }'
