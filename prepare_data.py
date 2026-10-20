import os
import shutil

# 設定原始路徑（根據你目前的目錄結構）
TRAIN_SRC = './archive (1)/train'
TEST_SRC = './archive (1)/test'

# 設定 YOLO 專用新路徑
TARGET_DIR = './datasets/emotion_data'
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def convert_to_yolo():
    # 1. 建立目錄
    for split in ['train', 'val']:
        os.makedirs(os.path.join(TARGET_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DIR, split, 'labels'), exist_ok=True)

    # 2. 處理資料
    configs = [(TRAIN_SRC, 'train'), (TEST_SRC, 'val')]
    
    for src_root, split in configs:
        if not os.path.exists(src_root):
            print(f"找不到資料夾: {src_root}")
            continue
            
        for idx, emotion in enumerate(EMOTIONS):
            source_path = os.path.join(src_root, emotion)
            if not os.path.exists(source_path): continue
            
            print(f"正在整理 {split} 的 {emotion} 圖片...")
            for filename in os.listdir(source_path):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    new_name = f"{emotion}_{filename}"
                    # 複製圖片
                    shutil.copy(os.path.join(source_path, filename),
                                os.path.join(TARGET_DIR, split, 'images', new_name))
                    
                    # 產生標籤檔 (假設人臉佔滿整張圖)
                    label_path = os.path.join(TARGET_DIR, split, 'labels', new_name.rsplit('.', 1)[0] + '.txt')
                    with open(label_path, 'w') as f:
                        f.write(f"{idx} 0.5 0.5 1.0 1.0")

    print("\n✅ 資料整理完成！路徑：./datasets/emotion_data")

if __name__ == "__main__":
    convert_to_yolo()
import os
from collections import Counter

# 1. 你的標籤資料夾路徑與類別名稱列表
LABEL_DIR = "./datasets/emotion_data/train/labels"
CLASS_NAMES = ["開心", "難過", "生氣", "平靜", "驚訝", "厭惡", "焦慮"]  # 依據你的類別順序修改

counter = Counter()

# 2. 讀取所有 .txt 檔並統計類別數量
for filename in os.listdir(LABEL_DIR):
    if filename.endswith(".txt"):
        with open(os.path.join(LABEL_DIR, filename), "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    counter[class_id] += 1

# 3. 印出檢測結果
print("=== 資料集類別數量統計 ===")
for class_id, count in sorted(counter.items()):
    name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"
    print(f"[{class_id}] {name}: {count} 張/標註")

# 4. 判斷不平衡度
max_count = max(counter.values())
min_count = min(counter.values())
ratio = max_count / min_count if min_count > 0 else 0
print(f"\n最高與最低數量比例: {ratio:.2f} : 1")
if ratio > 3:
    print("⚠️ 警告：資料集存在嚴重失衡，建議立刻執行補齊！")