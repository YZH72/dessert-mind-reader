import os

# 使用你專案真正的資料集路徑
LABELS_DIR = "./datasets/emotion_data/train/labels"

# 類別名稱對應
CLASS_NAMES = {
    "0": "開心 (happy)",
    "1": "難過 (sad)",
    "2": "生氣 (angry)",
    "3": "平靜 (neutral)",
    "4": "驚訝 (surprise)",
    "5": "厭惡 (disgust)",
    "6": "焦慮 (fear)"
}

counts = {k: 0 for k in CLASS_NAMES.keys()}

if os.path.exists(LABELS_DIR):
    for file_name in os.listdir(LABELS_DIR):
        if file_name.endswith(".txt"):
            label_path = os.path.join(LABELS_DIR, file_name)
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    cls_id = lines[0].split()[0]
                    if cls_id in counts:
                        counts[cls_id] += 1

    print("\n" + "="*40)
    print(" 📊 最新資料集類別數量統計結果")
    print("="*40)
    for cls_id, count in counts.items():
        print(f" [{cls_id}] {CLASS_NAMES[cls_id]}: {count} 張/標註")

    valid_counts = [c for c in counts.values() if c > 0]
    if valid_counts:
        max_c = max(valid_counts)
        min_c = min(valid_counts)
        ratio = max_c / min_c
        print("-" * 40)
        print(f" 📈 最高與最低數量比例: {ratio:.2f} : 1")
    print("="*40 + "\n")
else:
    print(f"❌ 找不到路徑: {LABELS_DIR}")