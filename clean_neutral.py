import os
import random

# ✅ 使用完整的真實資料集路徑
IMAGES_DIR = "./datasets/emotion_data/train/images"
LABELS_DIR = "./datasets/emotion_data/train/labels"

# YOLO 類別 3 為「平靜 (neutral)」
NEUTRAL_CLASS_ID = "3"
TARGET_COUNT = 4000

# 1. 蒐集所有類別為 3 (平靜) 的 txt 標籤檔
neutral_files = []
for file_name in os.listdir(LABELS_DIR):
    if file_name.endswith(".txt"):
        label_path = os.path.join(LABELS_DIR, file_name)
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if lines and lines[0].startswith(NEUTRAL_CLASS_ID + " "):
                neutral_files.append(file_name)

total_neutral = len(neutral_files)
print(f"📊 找到平靜類別標籤檔共: {total_neutral} 張")

if total_neutral > TARGET_COUNT:
    delete_count = total_neutral - TARGET_COUNT
    print(f"✂️ 需要刪除 {delete_count} 張平靜資料以達到目標 {TARGET_COUNT} 張...")
    
    files_to_delete = random.sample(neutral_files, delete_count)
    
    deleted_success = 0
    for txt_file in files_to_delete:
        base_name = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(LABELS_DIR, txt_file)
        
        img_path = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
            possible_path = os.path.join(IMAGES_DIR, base_name + ext)
            if os.path.exists(possible_path):
                img_path = possible_path
                break
        
        try:
            if os.path.exists(txt_path):
                os.remove(txt_path)
            if img_path and os.path.exists(img_path):
                os.remove(img_path)
            deleted_success += 1
        except Exception as e:
            print(f"刪除失敗 {base_name}: {e}")
            
    print(f"✅ 成功刪除 {deleted_success} 組平靜圖片與標籤！剩餘約 {TARGET_COUNT} 張。")
else:
    print("數量已符合或小於目標，無需刪除。")