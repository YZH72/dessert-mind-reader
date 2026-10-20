from ultralytics import YOLO

# 載入官方預訓練模型
model = YOLO('yolov8n.pt')

# 開啟相機偵測（source='0' 代表預設相機）
model.predict(source='0', show=True)