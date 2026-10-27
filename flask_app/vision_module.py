# vision_module.py （优化去闪烁版）
import cv2

class VisionModule:
    def __init__(self):
        self.prev_frame = None
        # 优化参数：提高阈值+增大最小面积，过滤细碎噪声闪烁
        self.params = {"threshold": 40, "min_area": 2000}

    def process(self, frame):
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return frame, {"motion_regions": []}

        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 加大高斯模糊，更强力去噪，减少摄像头噪声导致的闪烁
        prev_gray = cv2.GaussianBlur(prev_gray, (31, 31), 0)
        curr_gray = cv2.GaussianBlur(curr_gray, (31, 31), 0)

        diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(diff, self.params["threshold"], 255, cv2.THRESH_BINARY)
        
        # 形态学处理：先开运算去噪点，再闭运算合并区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.params["min_area"]:
                x, y, w, h = cv2.boundingRect(cnt)
                motion_regions.append((int(x), int(y), int(w), int(h)))
                # 运动检测框保持蓝色，和人脸框区分
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        self.prev_frame = frame.copy()
        return frame, {"motion_regions": motion_regions}
