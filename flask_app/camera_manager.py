# camera_manager.py
import cv2
import time
import threading
import queue
from picamera2 import Picamera2

class CameraManager:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None
        self.picam2 = None

        self.vision_module = None
        self.ai_module = None

        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self.latest_result = {}

    def set_vision_module(self, vision_module):
        self.vision_module = vision_module

    def set_ai_module(self, ai_module):
        self.ai_module = ai_module

    def start(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": (self.width, self.height)})
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("✅ 摄像头采集线程已启动")

    def _capture_loop(self):
        from config import config
        from db_helper import DBHelper
        db = DBHelper(config.DB_PATH)

        while self.running:
            try:
                frame_rgb = self.picam2.capture_array()
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                result = {}
                # 运动检测
                if self.vision_module is not None:
                    frame, vis_result = self.vision_module.process(frame)
                    result.update(vis_result)

                # 人脸识别
                if self.ai_module is not None and self.frame_count % config.DETECTION_INTERVAL == 0:
                    try:
                        ai_results = self.ai_module.predict(frame)
                        if ai_results and len(ai_results) > 0:
                            name, conf, box = ai_results[0]
                            result["name"] = name
                            result["confidence"] = conf
                    except Exception as e:
                        print(f"⚠️  AI推理异常: {e}")

                self.latest_result = result

                # FPS统计
                self.frame_count += 1
                self.fps_frame_count += 1
                now = time.time()
                if now - self.last_fps_time >= 1.0:
                    self.fps = self.fps_frame_count / (now - self.last_fps_time)
                    self.fps_frame_count = 0
                    self.last_fps_time = now

                # 自动存库（每60帧存一次）
                if self.frame_count % 60 == 0 and "name" in result:
                    try:
                        db.insert_detection("", result["name"], result.get("confidence", 0), result)
                    except Exception as e:
                        print(f"⚠️  存库异常: {e}")

                # 帧队列管理
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put((frame, result))

            except Exception as e:
                print(f"⚠️  采集异常: {e}")
                time.sleep(0.1)

    def get_frame(self):
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None, None

    def stop(self):
        self.running = False
        if self.picam2 is not None:
            self.picam2.stop()
        if self.thread is not None:
            self.thread.join(timeout=2)
        print("🛑 摄像头已停止")

    def get_status(self):
        return {
            "running": self.running,
            "fps": round(self.fps, 1),
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height
        }
