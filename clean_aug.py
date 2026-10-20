import os

IMAGE_DIR = "./datasets/emotion_data/train/images"
LABEL_DIR = "./datasets/emotion_data/train/labels"

deleted_img = 0
deleted_txt = 0

# 刪除所有 aug_ 開頭的圖片
for filename in os.listdir(IMAGE_DIR):
    if filename.startswith("aug_"):
        os.remove(os.path.join(IMAGE_DIR, filename))
        deleted_img += 1

# 刪除所有 aug_ 開頭的標籤檔
for filename in os.listdir(LABEL_DIR):
    if filename.startswith("aug_"):
        os.remove(os.path.join(LABEL_DIR, filename))
        deleted_txt += 1

print(f"🧹 清理完成！共刪除 {deleted_img} 張擴增圖片與 {deleted_txt} 個標籤檔。")