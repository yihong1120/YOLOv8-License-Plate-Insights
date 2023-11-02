import os
import shutil
import time
from ultralytics import YOLO

if __name__=='__main__':
    train_path="./runs/detect/train"
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    model = YOLO("yolov8l.pt")
    print("開始訓練 .........")
    t1=time.time()
    model.train(data="./data.yaml", epochs=150, imgsz=640)
    t2=time.time()
    print(f'訓練花費時間 : {t2-t1}秒')#2131.11984705925秒, epochs : 121
    path=model.export()
    print(f'模型匯出路徑 : {path}')
