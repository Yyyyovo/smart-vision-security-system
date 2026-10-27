# collect_face.py 无GUI人脸自动采集
import os
import time
import cv2
from picamera2 import Picamera2

# 配置参数
SAVE_DIR = "face_dataset"
USER_NAME = "huangshuning"    # 可修改为你的名字
PHOTO_COUNT = 20           # 采集20张样本
COUNTDOWN = 3              # 倒计时3秒后开始

# 创建保存目录
user_dir = os.path.join(SAVE_DIR, USER_NAME)
os.makedirs(user_dir, exist_ok=True)

# 初始化摄像头
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

print(f"⏳ {COUNTDOWN}秒后开始采集，请正对摄像头...")
for i in range(COUNTDOWN, 0, -1):
    print(f"{i}...")
    time.sleep(1)

count = 0
print("🚀 开始采集，请勿大幅移动...")

while count < PHOTO_COUNT:
    frame_rgb = picam2.capture_array()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 检测到人脸才保存
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    
    if len(faces) > 0:
        save_path = os.path.join(user_dir, f"{USER_NAME}_{count+1}.jpg")
        cv2.imwrite(save_path, frame)
        count += 1
        print(f"✅ 已采集 {count}/{PHOTO_COUNT} 张")
        time.sleep(0.5)
    else:
        print("⚠️  未检测到人脸，请调整位置")
        time.sleep(0.5)

picam2.stop()
print(f"🎉 采集完成，共保存 {count} 张照片到 {user_dir}")
