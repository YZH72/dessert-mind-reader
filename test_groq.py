import os
from groq import Groq

# 請確保環境變數中有 GROQ_API_KEY，或直接在此處填入字串: Groq(api_key="your_api_key")
client = Groq(api_key="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")

try:
    models = client.models.list()
    print("=== 你目前 Groq API Key 可使用的模型清單 ===")
    for model in models.data:
        print(f"- {model.id}")
except Exception as e:
    print("查詢失敗，原因：", e)