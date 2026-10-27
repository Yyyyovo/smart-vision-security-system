# config.py
import os

class Config:
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = False

    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    DB_PATH = os.path.join(os.path.dirname(__file__), 'detections.db')
    JPEG_QUALITY = 80
    STREAM_FPS = 20

    # 调试阶段每帧都检测，确保识别结果实时更新
    DETECTION_INTERVAL = 1

config = Config()
