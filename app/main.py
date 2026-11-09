import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from groq import Groq
from contextlib import asynccontextmanager
from ultralytics import YOLO

# 引入 3D 甜點映射邏輯與全域空間追蹤器
from logic import EmotionSpaceTracker, EMOTION_3D_SPACE

tracker = EmotionSpaceTracker()

# 輕量載入 YOLO 模型到全域 state
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🎬 正在初始化 YOLO 表情辨識模型...")
    try:
        model_path = r'runs/detect/train-7/weights/best.pt'
        app.state.yolo_model = YOLO(model_path)
        print("✅ YOLO 影像辨識模型權重載入成功！")
    except Exception as e:
        print(f"❌ YOLO 模型載入失敗: {str(e)}")
    yield
    print("🛑 釋放模型資源。")

app = FastAPI(
    title="甜點讀心術 —— AI甜點顧問後端系統",
    description="基於情感分析與用戶偏好的自主生成式對話服務。",
    version="2.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

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
                return res_text
        except Exception as e:
            last_error = str(e)
            continue
            
    return f"【AI 甜點顧問正在閉關調配處方】目前 Groq 服務繁忙（{last_error}），但已為您比對出空間幾何最佳解！"

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

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

# 1. 渠道一：文字情緒分析 API 路由
@app.post("/api/analyze-sentiment")
def analyze_sentiment(payload: SentimentRequest):
    try:
        classifier_prompt = f"""
        你是一個情緒分析師，請分析以下這句話的情緒比例。
        使用者輸入：「{payload.user_text}」
        請嚴格返回一個 JSON 格式的字典，包含以下 7 種情緒的機率：happy, sad, angry, surprise, fear, disgust, neutral
        格式範例：{{"happy": 0.1, "sad": 0.7, "angry": 0.0, "surprise": 0.0, "fear": 0.2, "disgust": 0.0, "neutral": 0.0}}
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

請用極其溫柔、感性的口吻，為他撰寫一段約 200~300 字的對話，解釋為何第一名【{primary_dessert}】最懂他，並給他一個品嚐療癒小儀式。必須強力且明確推薦這三款甜點：{dessert_text_for_prompt}。不要標題。
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
            "coord_list": coords,
            "dessert_list": matched_desserts,
            "warm_dialogue": ai_reply,
            "warm_recommendation": ai_reply
        }
    except Exception as e:
        return {"status": "error", "message": f"呼叫文字大腦失敗: {str(e)}"}

# 2. 渠道二：影像表情辨識 API (接收照片、經 YOLO 辨識並導出 3D 座標)
@app.post("/api/analyze-emotion-image")
async def analyze_emotion_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "圖片解碼失敗，請重新拍照"}

        # 圖片品質與尺寸優化 (480x480) 降低伺服器 CPU 開銷
        h, w = img.shape[:2]
        if max(h, w) > 480:
            scale = 480 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        if hasattr(app.state, "yolo_model") and app.state.yolo_model is not None:
            results = app.state.yolo_model(img, conf=0.25, imgsz=320, verbose=False, device='cpu')
            if len(results) > 0 and len(results[0].boxes) > 0:
                top_box = results[0].boxes[0]
                cls_id = int(top_box.cls[0].item())
                detected_emotion = results[0].names[cls_id]
            else:
                detected_emotion = "neutral"
        else:
            detected_emotion = "neutral"

        # 對映科學文獻 3D 座標 (P, A, D)
        emotion_mapping = {
            "happy": [0.81, 0.51, 0.46],
            "sad": [-0.63, -0.27, -0.33],
            "angry": [-0.51, 0.59, 0.25],
            "surprise": [0.4, 0.67, -0.13],
            "fear": [-0.64, 0.6, -0.43],
            "disgust": [-0.6, 0.35, 0.11],
            "neutral": [0.0, 0.0, 0.0]
        }
        coord_list = emotion_mapping.get(detected_emotion.lower(), [0.0, 0.0, 0.0])

        return {
            "status": "success",
            "detected_emotion": detected_emotion.capitalize(),
            "coord_list": coord_list
        }
    except Exception as e:
        return {"status": "error", "message": f"影像分析管線發生異常: {str(e)}"}

# 3. 接收 3D 座標並生成 Groq AI 諮商師對話與甜點處方
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
2. 結合這 3 款甜點的口感、溫度、物理質地或風味防線（重點分析第一名 {f'【{primary_dessert}】'}，並順理成章提及另外兩款）。
3. 給他一個挑選並品嚐的療癒小儀式。

【嚴格規範】：
1. 必須強力且明確推薦這三款甜點：{dessert_text_for_prompt}，這是模型計算出的最佳解。
2. 全程使用繁體中文（台灣習慣用語），不要標題。
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