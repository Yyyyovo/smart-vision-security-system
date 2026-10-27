# app.py
from flask import Flask, Response, jsonify, request, render_template
import cv2
import time
from config import config
from camera_manager import CameraManager

app = Flask(__name__)
camera = CameraManager(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
detection_enabled = True


def generate_frames():
    while True:
        frame, _ = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
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
    status = camera.get_status()
    status['detection_enabled'] = detection_enabled
    status['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({"code": 0, "data": status})

@app.route('/api/latest')
def api_latest():
    return jsonify({"code": 0, "data": camera.latest_result if camera.latest_result else {}})

@app.route('/api/statistics')
def api_statistics():
    from db_helper import DBHelper
    db = DBHelper(config.DB_PATH)
    return jsonify({"code": 0, "data": db.get_statistics()})

@app.route('/api/history')
def api_history():
    limit = request.args.get('limit', 20, type=int)
    from db_helper import DBHelper
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
    frame, result = camera.get_frame()
    if frame is None:
        return jsonify({"code": 1, "message": "无法获取画面"})

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = f"captures/capture_{timestamp}.jpg"
    cv2.imwrite(save_path, frame)

    from db_helper import DBHelper
    db = DBHelper(config.DB_PATH)
    name = result.get("name", "unknown") if result else "unknown"
    conf = result.get("confidence", 0.0) if result else 0.0
    db.insert_detection(save_path, name, conf, result)

    return jsonify({"code": 0, "message": "抓拍成功", "path": save_path})


# 启动入口
if __name__ == '__main__':
    # 注入运动检测模块
    try:
        from vision_module import VisionModule
        vision = VisionModule()
        camera.set_vision_module(vision)
        print("✅ 运动检测模块加载成功")
    except Exception as e:
        print(f"⚠️  运动检测模块加载失败: {e}")

    # 注入人脸识别模块
    try:
        from ai_module_face import FaceAI
        ai_face = FaceAI()
        camera.set_ai_module(ai_face)
    except Exception as e:
        print(f"⚠️  人脸识别模块加载失败: {e}")

    # 启动摄像头
    camera.start()

    print(f"🚀 服务启动，访问地址: http://0.0.0.0:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, threaded=True, debug=config.DEBUG)
