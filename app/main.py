import os
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from groq import Groq
from pydantic import BaseModel

from logic import EmotionSpaceTracker

app = FastAPI(
    title="甜點讀心術 —— AI甜點顧問後端系統",
    description="基於情感分析與用戶偏好的自主生成式對話服務。",
    version="1.0.0",
)

# 初始化 Groq 客戶端
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
# 依清單順序呼叫 Groq 模型，並記錄每次呼叫耗時。
def generate_groq_response(messages, temperature=0.7, max_tokens=1200, is_json=False, fallback=None):
    if client is None:
        if fallback is not None:
            return fallback
        raise RuntimeError("Render 尚未設定 GROQ_API_KEY。")

    # 根據你的 test_groq.py 實際授權清單進行自動備援排序
    candidate_models = [
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b"
    ]
    
    last_error = ""
    for model_name in candidate_models:
        started_at = perf_counter()
        try:
            kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
                "timeout": 4.0  # ✅ 設定 4 秒強刷 Timeout，防止 API 卡死
            }
            if model_name.startswith("openai/gpt-oss-"):
                kwargs["reasoning_effort"] = "low"
            elif model_name == "qwen/qwen3.8-27b":
                kwargs["reasoning_effort"] = "none"
            if is_json:
                kwargs["response_format"] = {"type": "json_object"}
                
            completion = client.chat.completions.create(**kwargs)
            choice = completion.choices[0]
            res_text = (choice.message.content or "").strip()
            if choice.finish_reason != "stop" or not res_text:
                last_error = f"回覆未完整結束（{choice.finish_reason}）"
                print(f"⚠️ [{model_name}] {last_error}")
                continue
            if not is_json and (
                not res_text.endswith(("。", "！", "？", "。」", "！」", "？」", ".", "!", "?"))
                or res_text.count("【") != res_text.count("】")
            ):
                last_error = "回覆缺少完整句尾或甜點名稱"
                print(f"⚠️ [{model_name}] {last_error}")
                continue
            return res_text
        except Exception as e:
            last_error = str(e)
            print(f"⚠️ 模型 [{model_name}] 呼叫失敗，原因: {last_error}")
            continue
        finally:
            print(f"[計時] Groq {model_name} {(perf_counter() - started_at) * 1000:.0f} ms")

    print(f"❌ 所有模型皆呼叫失敗，最後錯誤: {last_error}")
    if fallback is not None:
        return fallback
    raise RuntimeError("情緒分析服務目前未能完成回覆，請稍後重試。")


def fallback_dessert_dialogue(desserts):
    if not desserts:
        return "目前沒有符合你所選條件的甜點。你可以查看口味偏好設定，再試一次。"
    names = "、".join(f"【{name}】" for name in desserts)
    return f"依照你的設定，這次為你挑出{names}。如果想先試一款，可以從【{desserts[0]}】開始，留一點時間慢慢品嚐。"


# 沿用原專案的資料夾配置，templates 與 static 都在專案根目錄。
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static"), check_dir=False), name="static")

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
        
        cls_response_text = generate_groq_response(
            messages=[{"role": "user", "content": classifier_prompt}],
            temperature=0.1,
            is_json=True
        )
        
        import json
        # 解析 AI 吐回來的各情緒比例字典
        emotion_weights = json.loads(cls_response_text)
        
        # Step 2: 連結 logic.py 權重，把比例塞進 tracker.counts
        tracker = EmotionSpaceTracker()
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
        text_generation_prompt = f"""
使用者 {payload.username} 的心情：「{payload.user_text}」
分析結果為 {detected_emotion.capitalize()}，情緒座標（愉悅度、喚醒度、敏感度）為
{coords[0]:.2f}、{coords[1]:.2f}、{coords[2]:.2f}。
依推薦順序排列的甜點為：{dessert_text_for_prompt or "無合適甜點"}。

請用台灣繁體中文直接寫約 120～160 字的溫暖對話。先回應使用者的具體情境，再用風味或口感解釋第一名為何適合，簡短提及清單裡其餘選項，並給一個品嚐方式。
只推薦清單內的甜點，名稱用【】標示，並用完整句子結尾。清單為空時，直接說明沒有符合條件的推薦。
"""
        ai_reply = generate_groq_response(
            messages=[
                {"role": "system", "content": "你是一位溫柔且精通情感運算的 AI 甜點心理顧問。"},
                {"role": "user", "content": text_generation_prompt}
            ],
            temperature=0.7,
            fallback=fallback_dessert_dialogue(matched_desserts)
        )
        #ai_reply = completion.choices[0].message.content.strip()
        
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
def apply_image_scores(tracker, scores):
    # YOLO 分數不是情緒強度。未確定的部分放在 neutral，
    # 讓低分的辨識結果更靠近情緒空間中心。
    if not scores or any(
        emotion not in tracker.counts or not isfinite(score) or not 0 <= score <= 1
        for emotion, score in scores.items()
    ):
        raise ValueError("影像表情分數格式錯誤。")

    strongest_emotion = max(scores, key=scores.get)
    strongest_score = scores[strongest_emotion]
    if strongest_score <= 0.25:
        raise ValueError("影像辨識分數未達門檻，請重新拍照。")

    total = sum(scores.values())
    tracker.counts.update({
        emotion: score / total * strongest_score
        for emotion, score in scores.items()
    })
    tracker.counts["neutral"] += 1 - strongest_score
    return strongest_emotion


class CoordRecommendationRequest(BaseModel):
    coordinates: Optional[List[float]] = None
    detected_emotion: Optional[str] = None
    emotion_scores: Optional[Dict[str, float]] = None
    allergens: Optional[List[str]] = []     # 新增：過敏原列表
    preferences: Optional[List[str]] = []   # 新增：口味偏好列表

@app.post("/api/recommend-dessert-from-coord")
def recommend_dessert_from_coord(request: CoordRecommendationRequest):
    try:
        tracker = EmotionSpaceTracker()
        if request.emotion_scores is not None:
            emotion = apply_image_scores(tracker, request.emotion_scores)
            coords = tracker.calculate_user_coordinate().tolist()
        else:
            # 舊版呼叫端仍可傳單一類別或類別加座標。
            emotion = (request.detected_emotion or "").strip().lower()
            if emotion not in tracker.counts:
                return {"status": "error", "message": f"未設定的情緒類別：{emotion}"}

            tracker.counts[emotion] = 1
            coords = request.coordinates
            if coords is None:
                coords = tracker.calculate_user_coordinate().tolist()
            if len(coords) != 3 or not all(isfinite(value) for value in coords):
                return {"status": "error", "message": "情緒座標必須有三個有效數值。"}

        # 推薦與顯示使用同一組座標。
        matched_desserts = tracker.get_top_k_recommendations(
            user_vector=coords,
            user_allergens=request.allergens,
            user_preferences=request.preferences,
            k=3
        )
        
        # 為了讓 Prompt 更好閱讀，把 3 種甜點組合成字串（例如：【A】、【B】與【C】）
        dessert_text_for_prompt = "、".join([f"【{d}】" for d in matched_desserts])
        prompt = f"""
影像辨識結果為 {emotion.capitalize()}，情緒座標（愉悅度、喚醒度、敏感度）為
{coords[0]:.2f}、{coords[1]:.2f}、{coords[2]:.2f}。
依推薦順序排列的甜點為：{dessert_text_for_prompt or "無合適甜點"}。

請用台灣繁體中文直接寫約 120～160 字的溫暖對話。以「如果你今天覺得……」的方式回應可能的心情，再用風味或口感解釋第一名為何適合，簡短提及清單裡其餘選項，並給一個品嚐方式。
只推薦清單內的甜點，名稱用【】標示，並用完整句子結尾。清單為空時，直接說明沒有符合條件的推薦。
"""

        # 呼叫 Groq 大腦
        warm_text = generate_groq_response( 
            messages=[
                {"role": "system", "content": "你是一位溫柔且精通情感運算的 AI 甜點心理顧問。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            fallback=fallback_dessert_dialogue(matched_desserts)
        )
        
        #warm_text = completion.choices[0].message.content.strip()
        
        # 把這 3 個甜點的陣列也一起回傳給前端，方便前端動態生成 3 個卡片
        return {
            "status": "success", 
            "detected_emotion": emotion.capitalize(),
            "coord_list": [float(value) for value in coords],
            "warm_recommendation": warm_text,
            "warm_dialogue": warm_text,  # 雙重防呆
            "dessert_list": matched_desserts  # 這裡把 ["甜點1", "甜點2", "甜點3"] 丟給網頁
        }
        
    except Exception as e:
        return {"status": "error", "message": f"後端大腦錯誤: {str(e)}"}