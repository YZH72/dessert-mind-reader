import os
import cv2
import random
import numpy as np

# 1. 路徑設定
IMAGE_DIR = "./datasets/emotion_data/train/images"
LABEL_DIR = "./datasets/emotion_data/train/labels"

# 2. 定義純 OpenCV 影像擴增函式 (避免調用 albumentations 觸發 torch dll 問題)
def augment_image(image):
    h, w, _ = image.shape
    
    # 隨機調整亮度與對比度
    if random.random() > 0.3:
        alpha = random.uniform(0.8, 1.2) # 對比度
        beta = random.randint(-20, 20)    # 亮度
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
    # 隨機微幅旋轉 (-8 度到 +8 度)
    if random.random() > 0.4:
        angle = random.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
    # 隨機高斯模糊 (模擬不同鏡頭對焦)
    if random.random() > 0.7:
        ksize = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        
    # 隨機水平翻轉 (左右反轉)
    do_flip = False
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        do_flip = True
        
    return image, do_flip

# 3. 搜尋所有屬於 [1] 難過 的檔案
sad_files = []
for filename in os.listdir(LABEL_DIR):
    if filename.endswith(".txt"):
        txt_path = os.path.join(LABEL_DIR, filename)
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if parts and parts[0] == '1': # Class 1 是難過
                    base_name = os.path.splitext(filename)[0]
                    img_path = os.path.join(IMAGE_DIR, base_name + ".jpg")
                    if not os.path.exists(img_path):
                        img_path = os.path.join(IMAGE_DIR, base_name + ".png")
                    
                    if os.path.exists(img_path):
                        sad_files.append((img_path, txt_path, base_name))
                    break

print(f"找到 {len(sad_files)} 個『難過』類別的原始樣本。開始使用純 OpenCV 擴增...")

# 4. 開始擴增 (每張原始圖片擴增 8 張，約生成 3,400 張新數據)
TARGET_AUG_TIMES = 8
generated_count = 0

for img_path, txt_path, base_name in sad_files:
    image = cv2.imread(img_path)
    if image is None:
        continue
    
    # 讀取原本的標籤內容
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for i in range(TARGET_AUG_TIMES):
        aug_img, do_flip = augment_image(image.copy())
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts:
                cls_id = parts[0]
                x_center, y_center, w_val, h_val = map(float, parts[1:])
                
                # 如果做了水平翻轉，YOLO 的 x_center 座標需翻轉 (1 - x_center)
                if do_flip:
                    x_center = 1.0 - x_center
                    
                new_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_val:.6f} {h_val:.6f}\n")
                
        # 儲存擴增後的圖片與標籤
        new_img_name = f"aug_{i}_{base_name}.jpg"
        new_txt_name = f"aug_{i}_{base_name}.txt"
        
        cv2.imwrite(os.path.join(IMAGE_DIR, new_img_name), aug_img)
        with open(os.path.join(LABEL_DIR, new_txt_name), 'w') as f:
            f.writelines(new_lines)
            
        generated_count += 1

print(f"✅ 擴增完成！成功生成 {generated_count} 張新的『難過』表情數據與標籤！")