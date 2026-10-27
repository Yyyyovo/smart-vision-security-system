# main.py
"""
智能视觉安防系统 统一主程序
整合：摄像头采集 + 运动检测 + LBPH人脸识别 + 数据库 + Web服务
多线程生产者-消费者架构
"""
import cv2
import time
import threading
import queue
import signal
import sys
from picamera2 import Picamera2
from flask import Flask, Response, jsonify, request, render_template

from config import config
from db_helper import DBHelper

# ============================================================
# 全局变量
# ============================================================
app = Flask(__name__)
frame_queue = queue.Queue(maxsize=2)
running = True
detection_enabled = True

# 帧计数与FPS统计
frame_count = 0
fps = 0
last_fps_time = time.time()
fps_frame_count = 0

# 摄像头对象
picam2 = None

# 视觉处理模块
vision_module = None

# AI推理模块（安防组：人脸识别）
ai_module = None


# ============================================================
# 摄像头采集线程（生产者）
# ============================================================
def camera_thread():
    """摄像头采集线程：持续采集帧，处理后放入队列"""
    global picam2, frame_count, fps, last_fps_time, fps_frame_count
    global latest_result

    # 初始化摄像头
    picam2 = Picamera2()
    cam_config = picam2.create_preview_configuration(
        main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)}
    )
    picam2.configure(cam_config)
    picam2.start()
    time.sleep(2)
    print("[摄像头] 采集线程已启动")

    latest_result = {}
    db = DBHelper(config.DB_PATH)

    while running:
        try:
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            result = {}
            # 1. 运动检测
            if vision_module is not None:
                processed, vis_result = vision_module.process(frame_bgr)
                result.update(vis_result)
            else:
                processed = frame_bgr

            # 2. 人脸识别（每1帧检测一次，保证实时性）
            if ai_module is not None and frame_count % config.DETECTION_INTERVAL == 0:
                try:
                    ai_results = ai_module.predict(processed)
                    if ai_results and len(ai_results) > 0:
                        name, conf, box = ai_results[0]
                        result["name"] = name
                        result["confidence"] = conf
                except Exception as e:
                    print(f"[AI] 推理异常: {e}")

            latest_result = result

            # 3. 自动存库（每60帧存一次）
            if frame_count % 60 == 0 and "name" in result:
                try:
                    db.insert_detection("", result["name"], result.get("confidence", 0), result)
                except Exception as e:
                    print(f"[数据库] 存库异常: {e}")

            # 4. FPS统计
            frame_count += 1
            fps_frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = fps_frame_count / (now - last_fps_time)
                fps_frame_count = 0
                last_fps_time = now

            # 5. 放入队列（满了丢弃旧帧，保证实时性）
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put((processed, result))

        except Exception as e:
            print(f"[摄像头] 采集异常: {e}")
            time.sleep(0.1)

    picam2.stop()
    print("[摄像头] 采集线程已停止")


# ============================================================
# Flask Web服务（消费者）
# ============================================================
def generate_frames():
    """MJPEG视频流生成器"""
    while running:
        try:
            frame, _ = frame_queue.get(timeout=1.0)
        except queue.Empty:
            time.sleep(0.05)
            continue

        ret, jpeg = cv2.imencode('.jpg', frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(1.0 / config.STREAM_FPS)


# 页面路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# 视频流接口
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# 数据API
@app.route('/api/status')
def api_status():
    return jsonify({"code": 0, "data": {
        "running": running,
        "fps": round(fps, 1),
        "frame_count": frame_count,
        "detection_enabled": detection_enabled,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }})

@app.route('/api/latest')
def api_latest():
    try:
        _, result = frame_queue.get(timeout=0.5)
        return jsonify({"code": 0, "data": result})
    except queue.Empty:
        return jsonify({"code": 0, "data": {}})

@app.route('/api/statistics')
def api_statistics():
    db = DBHelper(config.DB_PATH)
    return jsonify({"code": 0, "data": db.get_statistics()})

@app.route('/api/history')
def api_history():
    limit = request.args.get('limit', 20, type=int)
    db = DBHelper(config.DB_PATH)
    return jsonify({"code": 0, "data": db.get_history(limit)})


# 控制API
@app.route('/api/start', methods=['POST'])
def api_start():
    global detection_enabled
    detection_enabled = True
    return jsonify({"code": 0, "message": "检测已开启"})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    global detection_enabled
    detection_enabled = False
    return jsonify({"code": 0, "message": "检测已暂停"})

@app.route('/api/capture', methods=['POST'])
def api_capture():
    try:
        frame, result = frame_queue.get(timeout=1.0)
    except queue.Empty:
        return jsonify({"code": 1, "message": "无法获取画面"})

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = f"captures/capture_{timestamp}.jpg"
    cv2.imwrite(save_path, frame)

    db = DBHelper(config.DB_PATH)
    name = result.get("name", "unknown") if result else "unknown"
    conf = result.get("confidence", 0.0) if result else 0.0
    db.insert_detection(save_path, name, conf, result)

    return jsonify({"code": 0, "message": "抓拍成功", "path": save_path})


# ============================================================
# 主程序入口
# ============================================================
def signal_handler(sig, frame):
    """处理Ctrl+C信号，优雅退出"""
    global running
    print("\n[系统] 收到退出信号，正在关闭...")
    running = False
    time.sleep(1)
    sys.exit(0)


def main():
    global vision_module, ai_module

    # 注册退出信号
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 1. 加载运动检测模块
    try:
        from vision_module import VisionModule
        vision_module = VisionModule()
        print("[系统] 运动检测模块已加载")
    except Exception as e:
        print(f"[系统] 运动检测模块加载失败: {e}")

    # 2. 加载人脸识别模块（安防组）
    try:
        from ai_module_face import FaceAI
        ai_module = FaceAI()
        print("[系统] 人脸识别模块已加载")
    except Exception as e:
        print(f"[系统] 人脸识别模块加载失败: {e}")

    # 3. 启动摄像头采集线程
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()

    # 等待摄像头就绪
    time.sleep(3)

    # 4. 启动Flask Web服务（主线程）
    print(f"[系统] Web服务启动，访问地址: http://0.0.0.0:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, threaded=True, debug=False)


if __name__ == '__main__':
    main()
