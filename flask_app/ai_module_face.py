# ai_module_face.py （优化检测率版）
import cv2
import json
import os

class FaceAI:
    def __init__(self, model_path="face_model.yml", names_path="face_names.json"):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # 加载人脸模型
        try:
            if os.path.exists(model_path):
                self.recognizer.read(model_path)
                print("✅ 人脸模型加载成功")
            else:
                print("⚠️  人脸模型不存在，训练后识别功能生效")
        except Exception as e:
            print(f"⚠️  人脸模型加载失败: {e}")
        
        # 加载姓名映射
        try:
            with open(names_path, 'r', encoding='utf-8') as f:
                self.names = json.load(f)
        except Exception:
            self.names = {}
        
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # LBPH置信度阈值：数值越小匹配越严格，大于阈值判定为陌生人
        self.threshold = 100

    def predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 回调参数：平衡检测率和误检
        # scaleFactor=1.1 精细缩放，保证不漏检；minNeighbors=5 过滤基础误检
        # 加 maxSize 限制，避免整幅画面被误判成人脸
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(80, 80),
            maxSize=(350, 350)
        )
        
        # 只保留面积最大的1个人脸，过滤多个小误检框
        if len(faces) > 1:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            faces = [faces[0]]

        results = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (100, 100))
            label, confidence = self.recognizer.predict(face_resized)
            
            if confidence < self.threshold:
                name = self.names.get(str(label), "Unknown")
            else:
                name = "Stranger"
            
            results.append((name, float(confidence), (int(x), int(y), int(w), int(h))))
            
            # 画面标注：熟人绿色，陌生人红色
            color = (0, 255, 0) if name != "Stranger" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} {confidence:.1f}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return results
