# train_face.py LBPH人脸识别模型训练
import cv2
import os
import json
import numpy as np

DATASET_DIR = "face_dataset"
MODEL_PATH = "face_model.yml"
NAMES_PATH = "face_names.json"

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_face_data():
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        label_map[label_id] = person_name
        print(f"🔍 正在处理：{person_name}")

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_rects = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            
            for (x, y, w, h) in face_rects:
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (100, 100))
                faces.append(face_resized)
                labels.append(label_id)
        
        label_id += 1

    return faces, np.array(labels), label_map

if __name__ == "__main__":
    print("🚀 开始训练人脸模型...")
    face_list, label_list, name_map = get_face_data()

    if len(face_list) == 0:
        print("❌ 未找到有效人脸样本，请先运行采集脚本")
        exit()

    print(f"📊 有效样本：{len(face_list)} 张，人物数：{len(name_map)}")
    
    recognizer.train(face_list, label_list)
    recognizer.save(MODEL_PATH)

    with open(NAMES_PATH, 'w', encoding='utf-8') as f:
        json.dump(name_map, f, ensure_ascii=False, indent=2)

    print(f"✅ 模型已保存：{MODEL_PATH}")
    print(f"✅ 姓名映射已保存：{NAMES_PATH}")
