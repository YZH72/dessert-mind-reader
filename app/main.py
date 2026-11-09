import os
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from groq import Groq

# 引入 3D 甜點映射邏輯與全域空間追蹤器
from logic import EmotionSpaceTracker, EMOTION_3D_SPACE

tracker = EmotionSpaceTracker()

app = FastAPI(
    title="甜點讀心術 —— AI甜點顧問後端系統",
    description="基於情感分析與用戶偏好的自主生成式對話服務。",
    version="2.0.0"
)

# 1. ✅ 新增 static 掛載：讓前端瀏覽器可以直接存取 static/best.onnx 模型檔
app.mount("/static", StaticFiles(directory="static"), name="static")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# 初始化 Groq 客戶端
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

# 萬能 Groq 呼叫函式：自動備援排序與 4 秒 Timeout 機制
def generate_groq_response(messages, temperature=0.7, max_tokens=600, is_json=False):
    candidate_models = [
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b"
    ]
    
    last_error = ""
    for model_name in candidate_models:
        try:
            kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": 4.0
            }
            if is_json:
                kwargs["response_format"] = {"type": "json_object"}
                
            completion = client.chat.completions.create(**kwargs)
            res_text = completion.choices[0].message.content.strip()
            if res_text:
                print(f"✅ 成功使用模型 [{model_name}] 生成對話內容！")
                return res_text
        except Exception as e:
            last_error = str(e)
            print(f"⚠️ 模型 [{model_name}] 呼叫失敗，原因: {last_error}")
            continue
            
    print(f"❌ 所有模型皆呼叫失敗，最後錯誤: {last_error}")
    return f"【AI 甜點顧問正在閉關調配處方】目前 Groq 服務繁忙（{last_error}），但已為您比對出空間幾何最佳解！"

# Pydantic 模型定義
class SentimentRequest(BaseModel):
    user_text: str
    username: Optional[str] = "點心愛好者"
    allergens: Optional[List[str]] = []
    preferences: Optional[List[str]] = []

class CoordRecommendationRequest(BaseModel):
    coordinates: List[float]
    detected_emotion: str
    allergens: Optional[List[str]] = []
    preferences: Optional[List[str]] = []

# 首頁路由
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    try:
        return templates.TemplateResponse(request=request, name="index.html")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return HTMLResponse(
            content=f"""
            <html>
                <body style="font-family: monospace; padding: 20px; background: #fff5f5; color: #900;">
                    <h2>💥 網頁渲染崩潰原因：</h2>
                    <pre style="background: #fee; padding: 15px; border: 1px solid #fcc; border-radius: 5px;">{error_details}</pre>
                </body>
            </html>
            """,
            status_code=500
        )

# 渠道一：文字情緒分析 API 路由 (SenticNet + logic.py 運算)
@app.post("/api/analyze-sentiment")
def analyze_sentiment(payload: SentimentRequest):
    try:
        classifier_prompt = f"""
        你是一個情緒分析師，請分析以下這句話的情緒比例。
        使用者輸入：「{payload.user_text}」
        
        請嚴格返回一個 JSON 格式的字典，包含以下 7 種情緒的機率（值在 0 到 1 之間，總和必須為 1）：
        happy, sad, angry, surprise, fear, disgust, neutral
        
        格式範例（只輸出 JSON，不要任何其他文字）：
        {{"happy": 0.1, "sad": 0.7, "angry": 0.0, "surprise": 0.0, "fear": 0.2, "disgust": 0.0, "neutral": 0.0}}
        """
        
        cls_response_text = generate_groq_response(
            messages=[{"role": "user", "content": classifier_prompt}],
            temperature=0.1,
            is_json=True
        )
        
        import json
        emotion_weights = json.loads(cls_response_text)
        
        tracker.reset_space()
        for emo, weight in emotion_weights.items():
            if emo in tracker.counts:
                tracker.counts[emo] = weight
                
        detected_emotion = max(emotion_weights, key=emotion_weights.get)
        coords = tracker.calculate_user_coordinate().tolist() 
        
        matched_desserts = tracker.get_top_k_recommendations(
            user_allergens=payload.allergens,
            user_preferences=payload.preferences,
            k=3
        )
        dessert_text_for_prompt = "、".join([f"【{d}】" for d in matched_desserts])
        primary_dessert = matched_desserts[0] if matched_desserts else "無合適甜點"
        
        text_generation_prompt = f"""
你是一位專業的「AI 情感甜點顧問」。
使用者 {payload.username} 吐露了心聲：「{payload.user_text}」
經過文本情緒維度分析，其對應的情緒為 {detected_emotion.capitalize()}，3D 情感座標為：
X(愉悅度):{coords[0]:.2f}、Y(喚醒度):{coords[1]:.2f}、Z(敏感度):{coords[2]:.2f}。

根據歐幾里得幾何特徵空間運算，最能精準補償/療癒他當前身心狀態的前三名甜點選擇是：{dessert_text_for_prompt}。

請用極其溫柔、感性、的口吻，為他撰寫一段對話。
內容需包含：
1. 溫柔同理他當前文字所透露出的具體情境與焦慮感受（例如專題卡關、壓力和不安全感）。
2. 結合這 3 款甜點的口感、溫度、物理質地或風味防線（可以重點分析第一名 {f'【{primary_dessert}】'}，並順理成章地提及另外兩款選擇），完美解釋為什麼這三款甜點是此時此刻最懂他的存在。
3. 給他一個挑選並品嚐的療癒小儀式。

【嚴格規範】：
1. 必須強力且明確推薦這三款甜點：{dessert_text_for_prompt}，這是模型計算出的最佳解。
2. 全程使用繁體中文（台灣習慣用語），字數約 200~300 字。
3. 不要標題，直接輸出溫暖的對話內容。
"""

        ai_reply = generate_groq_response(
            messages=[
                {"role": "system", "content": "你是一位溫柔且精通情感運算的 AI 甜點心理顧問。"},
                {"role": "user", "content": text_generation_prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        
        return {
            "status": "success",
            "detected_emotion": detected_emotion.capitalize(),
            "detected_emotion_lower": detected_emotion.lower(),
            "emotionLabel": detected_emotion.capitalize(),
            "emotion_label": detected_emotion.capitalize(),
            "coord_list": coords,
            "coordinates": coords,
            "dessert_list": matched_desserts,
            "warm_dialogue": ai_reply,
            "warm_recommendation": ai_reply,
            "user_input": payload.user_text
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"呼叫文字大腦失敗了，原因：{str(e)}"
        }

# 渠道二：接收前端 ONNX 推論出的 3D 座標並生成 AI 對話
@app.post("/api/recommend-dessert-from-coord")
async def recommend_dessert_from_coord(request: CoordRecommendationRequest):
    try:
        tracker.reset_space()
        emo_key = request.detected_emotion.lower()
        if emo_key in tracker.counts:
            tracker.counts[emo_key] = 1
            
        matched_desserts = tracker.get_top_k_recommendations(
            user_allergens=request.allergens,
            user_preferences=request.preferences,
            k=3
        )
        
        dessert_text_for_prompt = "、".join([f"【{d}】" for d in matched_desserts])
        primary_dessert = matched_desserts[0] if matched_desserts else "無合適甜點"
        
        prompt = f"""
你是一位專業的「AI 情感甜點顧問」。
使用者目前面部微表情經 YOLO 辨識為 {request.detected_emotion}，映射之 3D 情感座標為：
X(愉悅度):{request.coordinates[0]:.2f}、Y(喚醒度):{request.coordinates[1]:.2f}、Z(敏感度):{request.coordinates[2]:.2f}。

根據歐幾里得幾何特徵空間運算，最能精準補償/療癒他當前身心狀態的前三名甜點選擇是：{dessert_text_for_prompt}。

請用極其溫柔、感性、深夜療癒咖啡館顧問的口吻，為他撰寫一段 200~250 字的對話。
內容需包含：
1. 溫柔同理他當前表情所透露出的身心狀態。
2. 結合這 3 款甜點的口感、溫度、物理質地或風味防線（可以重點分析第一名 {f'【{primary_dessert}】'}，並順理成章地提及另外兩款選擇），完美解釋為什麼這三款甜點是此時此刻最懂他的存在。
3. 給他一個挑選並品嚐的療癒小儀式。

【嚴格規範】：
1. 必須強力且明確推薦這三款甜點：{dessert_text_for_prompt}，這是模型計算出的最佳解。
2. 全程使用繁體中文（台灣習慣用語）。
3. 不要標題，直接輸出溫暖的對話內容。
"""

        warm_text = generate_groq_response(
            messages=[
                {"role": "system", "content": "你是一位溫柔且精通情感運算的 AI 甜點心理顧問。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        
        return {
            "status": "success", 
            "warm_recommendation": warm_text,
            "warm_dialogue": warm_text,
            "dessert_list": matched_desserts
        }
        
    except Exception as e:
        return {"status": "error", "message": f"後端大腦錯誤: {str(e)}"}