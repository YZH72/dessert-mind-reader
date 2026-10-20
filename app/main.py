import os
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional,List
from groq import Groq
from contextlib import asynccontextmanager  # 用於管理 YOLO 生命週期
from ultralytics import YOLO                # 用於載入 YOLO 模型
import cv2
import numpy as np
# 引入你的 3D 甜點映射邏輯
from logic import EmotionSpaceTracker, EMOTION_3D_SPACE

# 初始化全域的空間追蹤器
tracker = EmotionSpaceTracker()

# 定義生命週期管理，讓網頁伺服器啟動時，自動載入 YOLO 模型到記憶體中
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🎬 正在初始化 YOLO 表情辨識模型...")
    try:
        # 指向你 train-7 的最强大腦權重檔
        model_path = r'runs/detect/train-7/weights/best.pt'
        # 載入並儲存至 app.state，這樣未來的 API 路由都可以直接呼叫
        app.state.yolo_model = YOLO(model_path)
        print("✅ YOLO 影像辨識模型權重載入成功！")
    except Exception as e:
        print(f"❌ YOLO 模型載入失敗，錯誤訊息: {str(e)}")
    yield
    print("🛑 伺服器關閉，釋放模型資源。")

# 將生命週期 lifespan 注入 FastAPI 實例
app = FastAPI(
    title="甜點讀心術 —— AI甜點顧問後端系統",
    description="基於情感分析與用戶偏好的自主生成式對話服務。",
    version="1.0.0",
    lifespan=lifespan
)

# 初始化 Groq 客戶端
client = Groq(api_key="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")

# 直接把你的 Windows 實際路徑複製貼上
# ✅ 改成相對路徑（動態抓取專案目錄下的 templates 資料夾）
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Pydantic 模型，用來定義前端送過來的 JSON 結構
class SentimentRequest(BaseModel):
    user_text: str
    username: Optional[str] = "點心愛好者"
    allergens: Optional[List[str]] = []     # 新增：過敏原列表
    preferences: Optional[List[str]] = []   # 新增：口味偏好列表

# 設定給開源大腦的「甜點諮商師說明書」
SYSTEM_PROMPT = """
你是一位精通「情緒維度與甜點特性映射」的心理諮商師兼頂級甜點師。
你的任務是根據使用者吐露的心情，分析其隱含的三維情緒指標（Valence 效價、Arousal 喚醒度、Dominance 支配度），並即時生成一段結合心理學、物理口感與感官療癒的溫暖對話。

請在心中依據以下三維核心理論進行映射（但不要在回答中死板地列出維度名稱，請將其內化為流暢的溫暖文字）：
1. Valence (效價 - 正/負向)：
   - 正向 (快樂/雀躍)：推薦明亮、水果酸甜、層次活潑的風味。
   - 負向 (難過/挫折/壓力)：推薦高飽足感、乳酪濃郁感或黑巧克力等能刺激多巴胺的風味。
2. Arousal (喚醒度 - 高/低)：
   - 高喚醒 (焦慮/暴躁/緊張)：大腦處於緊繃宣洩期。推薦「爽脆、有咬勁、具豐富層次物理回饋」的甜點，透過大量咀嚼轉移注意力。
   - 低喚醒 (低落/孤單/疲憊)：能量枯竭。推薦「溫熱、綿密、滑順、入口即化」的甜點，給予無負擔的溫柔包裹感。
3. Dominance (支配度/掌控感 - 高/低)：
   - 低支配 (無助/失控/被動)：使用者感到生活失控。推薦「外形結構精緻固定、需要特定品嚐儀式（如：用湯匙敲碎烤布蕾脆皮、親手切開塔皮）」的甜點，透過具備「掌控感」的動作重建心理秩序。
   - 高支配 (自信/主導)：推薦「流體、開放式、視覺張力強」的甜點。

請遵循以下三步驟結構撰寫回應（嚴禁出現步驟標題或任何標籤）：
- 情緒鏡像與同理：精準捕捉使用者當下的具體情境（如專題卡住、被檢討、生活焦慮），深度同理並接納對方的感受。
- 物理與心理學解釋：溫柔地告訴他，為什麼他現在的身心和大腦，會需要某種特定「口感、溫度、風味」的甜點（帶入上述三維映射的科學邏輯）。
- 甜點推薦與儀式：端出最完美的甜點，並引導他如何帶著儀式感去享受它，完成這場讀心療癒。

語氣規範：
- 必須完全使用繁體中文（台灣習慣用語），語氣要無比溫柔、親切、具備心理支持感。
- 嚴禁死板地背誦「Valence/Arousal/Dominance」術語，要將理論融合成長度適中的溫暖對話。
"""

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
                    <h2>💥 抓到你了！網頁渲染崩潰原因：</h2>
                    <pre style="background: #fee; padding: 15px; border: 1px solid #fcc; border-radius: 5px;">{error_details}</pre>
                </body>
            </html>
            """,
            status_code=500
        )

# 文字情緒分析 API 路由
@app.post("/api/analyze-sentiment")
def analyze_sentiment(payload: SentimentRequest):
    try:
        # Step 1: 讓 AI 改吐出 JSON 格式的「情緒百分比機率分佈」
        classifier_prompt = f"""
        你是一個情緒分析師，請分析以下這句話的情緒比例。
        使用者輸入：「{payload.user_text}」
        
        請嚴格返回一個 JSON 格式的字典，包含以下 7 種情緒的機率（值在 0 到 1 之間，總和必須為 1）：
        happy, sad, angry, surprise, fear, disgust, neutral
        
        格式範例（只輸出 JSON，不要任何其他文字）：
        {{"happy": 0.1, "sad": 0.7, "angry": 0.0, "surprise": 0.0, "fear": 0.2, "disgust": 0.0, "neutral": 0.0}}
        """
        
        cls_completion = client.chat.completions.create(
            model="qwen/qwen3.8-27b",  # 或使用 "llama-3.2-11b-vision-preview"
            messages=[{"role": "user", "content": classifier_prompt}],
            temperature=0.1,
            response_format={"type": "json_object"} # 強制 Groq 回傳標準 JSON
        )
        
        import json
        # 解析 AI 吐回來的各情緒比例字典
        emotion_weights = json.loads(cls_completion.choices[0].message.content.strip())
        
        # Step 2: 連結 logic.py 權重，把比例塞進 tracker.counts
        tracker.reset_space()
        for emo, weight in emotion_weights.items():
            if emo in tracker.counts:
                tracker.counts[emo] = weight # 權重
                
        # 找出主導情緒（比例最高的），用來維持後面 Prompt 裡的變數
        detected_emotion = max(emotion_weights, key=emotion_weights.get)
        
        # 呼叫 logic.py 的權重加總算法，算出來的 user_vector 就是複合情感重心
        coords = tracker.calculate_user_coordinate().tolist() 
        
        # 撈出真正融合多種情緒後，距離最近的前三名甜點
        matched_desserts = tracker.get_top_k_recommendations(
            user_allergens=payload.allergens,
            user_preferences=payload.preferences,
            k=3
        )
        dessert_text_for_prompt = "、".join([f"【{d}】" for d in matched_desserts])
        primary_dessert = matched_desserts[0] if matched_desserts else "無合適甜點"
        
        # 構建 Prompt（帶入偏好與過敏原資訊）
        allergens_info = "無" if not payload.allergens else "、".join(payload.allergens)
        prefs_info = "無特別偏好" if not payload.preferences else "、".join(payload.preferences)

        # Step 3: 將運算出來的「真實甜點結果」餵給大腦生成最終對話（比照影像路由的高規格）
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

        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b", # 文字也用大模型，生成的語氣會更像諮商師
            messages=[
                {"role": "system", "content": "你是一位溫柔且精通情感運算的 AI 甜點心理顧問。"},
                {"role": "user", "content": text_generation_prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        ai_reply = completion.choices[0].message.content.strip()
        
        # 🚀【關鍵成功因素】：這裡一定要回傳 dessert_list，前端網頁的 JavaScript 才能抓去顯示 Top 1！
       # 4. 🚀【超級防呆回傳】把所有前端可能會讀取的變數名稱、大小寫全部塞進去！
        # 這樣不管前端 JavaScript 寫 data.detected_emotion、data.emotionLabel 還是 data.coord_list，都絕對接得到！
        return {
            "status": "success",
            "detected_emotion": detected_emotion.capitalize(),   # 例如: "Sad", "Happy"
            "detected_emotion_lower": detected_emotion.lower(),   # 小寫防呆: "sad", "happy"
            "emotionLabel": detected_emotion.capitalize(),        # 欄位別名防呆
            "emotion_label": detected_emotion.capitalize(),       # 欄位別名防呆
            
            "coord_list": coords,                                 # [X, Y, Z] 陣列
            "coordinates": coords,                                # 欄位別名防呆
            
            "dessert_list": matched_desserts,                     # 甜點陣列
            "warm_dialogue": ai_reply,                            # 舊文字路由對話 Key
            "warm_recommendation": ai_reply,                      # 影像路由對話 Key
            "user_input": payload.user_text
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"呼叫文字大腦失敗了，原因：{str(e)}"
        }
# 串接 YOLO 與 3D 甜點空間的自拍分析 API 路由
@app.post("/api/analyze-emotion-image")
async def analyze_emotion_image(file: UploadFile = File(...)):
    try:
        # 1. 讀取前端傳過來的二進位照片資料
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "圖片解碼失敗，請重新拍照"}

        # 2. 呼叫記憶體中的 YOLO 模型進行表情推論
        if hasattr(app.state, "yolo_model") and app.state.yolo_model is not None:
            # 跑模型預測 (設定信心度門檻值，並關閉 verbose 輸出讓終端機乾淨)
            results = app.state.yolo_model(img, conf=0.25, verbose=False, device='cpu')
            
            # 判斷有沒有偵測到任何人臉表情
            if len(results) > 0 and len(results[0].boxes) > 0:
                # 抓出信心度最高的那個人臉框的類別 ID
                top_box = results[0].boxes[0]
                cls_id = int(top_box.cls[0].item())
                # 取得對應的情緒名稱 (例如 'happy', 'sad', 'angry' 等)
                detected_emotion = results[0].names[cls_id]
            else:
                # 如果拍照沒對準臉，或者背景太暗沒偵測到，則預設為 Neutral
                detected_emotion = "neutral"
        else:
            return {"status": "error", "message": "後端 YOLO 模型尚未成功初始化"}

        # 3. 🚀 【空間映射】安全查表或調用對應方法
        # 如果你的 tracker 內部是用字典儲存情緒座標 (例如 self.emotion_dict 或 self.coordinates)
        if hasattr(tracker, 'emotion_dict'):
            coord_list = tracker.emotion_dict.get(detected_emotion.lower(), [0.0, 0.0, 0.0])
        elif hasattr(tracker, 'get_coordinates'):
            coord_list = tracker.get_coordinates(detected_emotion)
        else:
            # 💡 防呆：如果完全對不上，根據偵測到的情緒給予科學文獻對應的 V, A, S 座標
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
        # 4. 回傳給前端網頁（注意：detected_emotion 首字母大寫，美化網頁顯示）
        return {
            "status": "success",
            "detected_emotion": detected_emotion.capitalize(),
            "coord_list": coord_list
        }

    except Exception as e:
        return {"status": "error", "message": f"影像分析管線發生異常: {str(e)}"}

# 前端第二步：接收 3D 座標並交由 Groq 大腦生成對話
class CoordRecommendationRequest(BaseModel):
    coordinates: List[float]
    detected_emotion: str
    allergens: Optional[List[str]] = []     # 新增：過敏原列表
    preferences: Optional[List[str]] = []   # 新增：口味偏好列表

@app.post("/api/recommend-dessert-from-coord")
async def recommend_dessert_from_coord(request: CoordRecommendationRequest):
    try:
        # 將前端送回的座標手動塞入 tracker 中，以調用距離運算
        tracker.reset_space()
        # 藉由直接操作暫時權重，將座標導回你的 3D 空間
        # 如果 request.detected_emotion 小寫在空間中，就給它計數 1
        emo_key = request.detected_emotion.lower()
        if emo_key in tracker.counts:
            tracker.counts[emo_key] = 1
            
        # k 為 3，撈出 3D 空間中距離目前情緒重心最近的前三名甜點
        matched_desserts = tracker.get_top_k_recommendations(
            user_allergens=request.allergens,
            user_preferences=request.preferences,
            k=3
        )
        
        # 為了讓 Prompt 更好閱讀，把 3 種甜點組合成字串（例如：【A】、【B】與【C】）
        dessert_text_for_prompt = "、".join([f"【{d}】" for d in matched_desserts])
        primary_dessert = matched_desserts[0] if matched_desserts else "無合適甜點" # 以第一名作為主推
        
        # 將運算結果餵給 Groq，確保它能同時對這 3 款甜點展開暖心分析
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

        # 呼叫 Groq 大腦
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b", 
            messages=[
                {"role": "system", "content": "你是一位溫柔且精通情感運算的 AI 甜點心理顧問。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        
        warm_text = completion.choices[0].message.content.strip()
        
        # 把這 3 個甜點的陣列也一起回傳給前端，方便前端動態生成 3 個卡片
        return {
            "status": "success", 
            "warm_recommendation": warm_text,
            "dessert_list": matched_desserts  # 這裡把 ["甜點1", "甜點2", "甜點3"] 丟給網頁
        }
        
    except Exception as e:
        return {"status": "error", "message": f"後端大腦錯誤: {str(e)}"}