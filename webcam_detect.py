from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont # 新增：引入 Pillow 用於處理中文
from logic import EmotionSpaceTracker

# 1. 載入剛跑完的「最強大腦」best.pt
model_path = r'runs/detect/train-7/weights/best.pt'
model = YOLO(model_path)

# 2. 初始化 3D 空間追蹤器與流程狀態
tracker = EmotionSpaceTracker()
is_locked = False  
top_desserts = []  
final_coord = np.zeros(3)
dominant_emotions = ""

# 3. 開啟電腦鏡頭
cap = cv2.VideoCapture(0)
# 【優化 1】設定鏡頭解析度為 640x480，顯著提升 I/O 與推論速度
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("錯誤：無法開啟鏡頭。")
    exit()

# ==========================================================
# 【優化 2】全域預載字型，避免在迴圈內反覆 IO 讀取字型檔
# ==========================================================
try:
    GLOBAL_FONT_16 = ImageFont.truetype("msjh.ttc", 16)
    GLOBAL_FONT_18 = ImageFont.truetype("msjh.ttc", 18)
    GLOBAL_FONT_22 = ImageFont.truetype("msjh.ttc", 22)
except IOError:
    GLOBAL_FONT_16 = ImageFont.load_default()
    GLOBAL_FONT_18 = ImageFont.load_default()
    GLOBAL_FONT_22 = ImageFont.load_default()

# ==========================================================
# 支援 OpenCV 畫面顯示中文的輔助函式
# ==========================================================
def draw_chinese_text_fast(img, text_list_with_props):
    """ 透過 Pillow 在 OpenCV 影像上繪製中文字型 """
    # 將 OpenCV 的 BGR 影像轉為 PIL 的 RGB 影像
    # [(text, pos, font, color), ...]
    """【優化 3】批次繪製中文字串，一次性完成 BGR <-> RGB 轉換，極大化效率"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(img_pil)
    
    for text, position, font, color in text_list_with_props:
        draw.text(position, text, font=font, fill=color)
    
    # 再轉回 OpenCV 的 BGR 影像格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


print("==== 3D 情感感官推薦系統已啟動 ====")
print(" 操作說明：")
print(" 【空白鍵 (Space)】：定格畫面，當場鎖定並輸出 Top 3 甜點推薦！")
print(" 【R 鍵 (Reset)】  ：清空目前的累計情緒，重新進行測驗。")
print(" 【Q 鍵 (Quit)】   ：關閉並退出程式。\n")

while True:
    if not is_locked:
        ret, frame = cap.read()
        if not ret:
            print("無法接收鏡頭影像，程式結束。")
            break
        frame = cv2.flip(frame, 1) # 鏡像翻轉，讓使用者看起來像在照鏡子（1 代表左右翻轉，0 代表上下翻轉）
        results = model.predict(source=frame, conf=0.25, imgsz=640, verbose=False, show=False) #預測

        for r in results:
            for c in r.boxes.cls:
                emotion_label = model.names[int(c)].lower()
                tracker.record_frame_emotion(emotion_label)

        display_frame = results[0].plot()

    # ========================================================================
    # 拓寬畫布！建立一個寬度增加 360 像素的右側大型資訊看板
    # ========================================================================
    h, w, c = display_frame.shape
    info_panel_width = 360
    # 建立一個全黑的右側面板
    info_panel = np.zeros((h, info_panel_width, c), dtype=np.uint8)
    # 給面板加上一點深夜科技感的深灰色科技邊框與微底色
    cv2.rectangle(info_panel, (0, 0), (info_panel_width, h), (20, 20, 20), -1)
    cv2.line(info_panel, (0, 0), (0, h), (0, 255, 0), 2) # 科技綠分割線

    # 將左邊的人臉影像與右邊的資訊面板水平拼接（拼起來總寬度變大）
    combined_frame = np.hstack((display_frame, info_panel))

    # --- 右側面板文字渲染 ---
    # 計算右側資訊起始的 X 座標位置 (即原本畫面的寬度再往右移)
    x_offset = w + 20

    if is_locked:
        # 定格鎖定狀態：在右側呈現完美的學術報告與甜點清單
        cv2.putText(combined_frame, "SWEET SENSE: ANALYSIS", (x_offset, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 127), 2, cv2.LINE_AA)
        
        text_batch = [
            (
                f"主要情緒: {dominant_emotions}",
                (x_offset, 70),
                GLOBAL_FONT_18,
                (0, 200, 255),
            ),
            (
                f"X : {final_coord[0]:.2f} (豐富度/熱量需求)",
                (x_offset, 105),
                GLOBAL_FONT_16,
                (180, 180, 180),
            ),
            (
                f"Y : {final_coord[1]:.2f} (質地/脆軟程度)",
                (x_offset, 130),
                GLOBAL_FONT_16,
                (180, 180, 180),
            ),
            (
                f"Z : {final_coord[2]:.2f} (風味/刺激強度)",
                (x_offset, 155),
                GLOBAL_FONT_16,
                (180, 180, 180),
            ),
            ("匹配推薦清單:", (x_offset, 200), GLOBAL_FONT_18, (255, 255, 255)),
        ]

        for i, dessert in enumerate(top_desserts, 1):
            text_batch.append(
                (
                    f"No.{i} {dessert}",
                    (x_offset, 235 + ((i - 1) * 45)),
                    GLOBAL_FONT_22,
                    (255, 255, 255),
                )
            )

        text_batch.append(
            (
                "[ 按 R 鍵重新拍攝 ]\n[ 按 Q 鍵退出程式 ]",
                (x_offset, 380),
                GLOBAL_FONT_16,
                (100, 100, 255),
            )
        )
        combined_frame = draw_chinese_text_fast(combined_frame, text_batch)
    else:
        # 即時動態偵測狀態：提示使用者可以按空白鍵
        cv2.putText(combined_frame, "REALTIME AI", (x_offset, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        lines = [
            "System Analyzing...",
            "Please face the camera.",
            "",
            "Press [ SPACE ] to Lock",
            "and view Dessert Menu.",
        ]
        for idx, line in enumerate(lines):
            cv2.putText(
                combined_frame,
                line,
                (x_offset, 100 + idx * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

    # 顯示拼接後、絕不遮臉的完美畫面
    cv2.imshow("Emotion Project - 3D Recommendation Space", combined_frame)

    # --- 鍵盤事件監聽 ---
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' ') and not is_locked:
        is_locked = True
        final_coord = tracker.calculate_user_coordinate()
        top_desserts = tracker.get_top_k_recommendations(k=3)
        current_weights = tracker.get_current_weights()
        # 取得情緒統計並找出最強的前兩個
        weights = tracker.get_current_weights()
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        dominant_emotions = f"{sorted_weights[0][0]}({int(sorted_weights[0][1]*100)}%) {sorted_weights[1][0]}({int(sorted_weights[1][1]*100)}%)"

        print("\n" + "="*50)
        print("🎉 【偵測成功！3D 情感感官推薦報告】 🎉")
        print(f" 當前表情權重累積: {current_weights}")
        print(f" 計算出之情緒重心座標: [X: {final_coord[0]:.3f}, Y: {final_coord[1]:.3f}, Z: {final_coord[2]:.3f}]")
        print("-"*50)
        print(" 💡 【為您精準匹配的 Top 3 熱門甜點菜單】:")
        for i, dessert in enumerate(top_desserts, 1):
            print(f"    ▶ 第 {i} 名：{dessert}")
        print("="*50 + "\n")

    elif key == ord('r'):
        is_locked = False
        tracker.reset_space()
        top_desserts = []
        print("🔄 系統已重設，請面向鏡頭重新開始分析。")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("程式已關閉。")