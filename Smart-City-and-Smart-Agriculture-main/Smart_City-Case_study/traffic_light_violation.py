import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import sqlite3
from utils.telegram import send_violation_alert

# --- CẤU HÌNH VẼ LINE ---
line_pts = []
def mouse_callback(event, x, y, flags, param):
    global line_pts
    if event == cv2.EVENT_LBUTTONDOWN and len(line_pts) < 2:
        line_pts.append((x, y))
        print(f"Point {len(line_pts)}: {(x, y)}")

VIDEO_SOURCE = r'C:\TPTMVsNNTM\Smart-City-and-Smart-Agriculture-main\Smart_City-Case_study\hi2.mp4'

# --- CHỌN VẠCH TRÊN FRAME ĐẦU TIÊN ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Không thể đọc video.")
cv2.namedWindow("Select Line")
cv2.setMouseCallback("Select Line", mouse_callback)
while True:
    disp = first_frame.copy()
    if len(line_pts) >= 1:
        cv2.circle(disp, line_pts[0], 5, (0, 255, 0), -1)
    if len(line_pts) == 2:
        cv2.line(disp, line_pts[0], line_pts[1], (0, 255, 0), 2)
    cv2.imshow("Select Line", disp)
    key = cv2.waitKey(1) & 0xFF
    if len(line_pts) == 2 and key == ord('s'):
        break
    if key == ord('q'):
        cap.release(); cv2.destroyAllWindows(); exit("Hủy chọn.")
cv2.destroyWindow("Select Line")
cap.release()

# --- TẢI MODEL ---
car_model = YOLO('yolov8n.pt')  # COCO: class 2 = car
tl_model = YOLO(r'C:\TPTMVsNNTM\Smart-City-and-Smart-Agriculture-main\Smart_City-Case_study\best_traffic_nano_yolo.pt')

os.makedirs('vi_pham', exist_ok=True)
os.makedirs('data', exist_ok=True)

track_history = {}
frame_count = 0

# Hàm kiểm tra phía của điểm so với đường thẳng
def side_of_line(pt, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)

# Hàm log vi phạm vào CSDL và gửi Telegram
def log_violation(tid, image_path):
    conn = sqlite3.connect('data/violations.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS violations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        car_id INTEGER,
                        image TEXT,
                        timestamp TEXT)''')
    cursor.execute("INSERT INTO violations (car_id, image, timestamp) VALUES (?, ?, ?)",
                   (tid, image_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    # Gửi telegram
    send_violation_alert(image_path, tid)

# --- STREAM với TRACKING & LIGHT DETECTION ---
for result in car_model.track(
        source=VIDEO_SOURCE,
        conf=0.5,
        iou=0.5,
        classes=[2],  # Chỉ phát hiện xe hơi
        persist=True,
        stream=True
    ):
    frame_count += 1
    frame = result.orig_img.copy()

    # 1) Vẽ vạch
    if len(line_pts) >= 2:
        cv2.line(frame, line_pts[0], line_pts[1], (0, 255, 0), 2)

    # 2) Detect đèn giao thông
    tl_res = tl_model(frame, conf=0.3)[0]
    tl_state = None
    for tl_box in tl_res.boxes:
        x1_l, y1_l, x2_l, y2_l = tl_box.xyxy.cpu().numpy().astype(int)[0]
        cls_id = int(tl_box.cls.cpu().item())
        conf_l = float(tl_box.conf.cpu().item())
        name = tl_model.model.names[cls_id]
        color = (0, 255, 0) if name == 'green' else (0, 0, 255) if name == 'red' else (255, 255, 0)
        cv2.rectangle(frame, (x1_l, y1_l), (x2_l, y2_l), color, 2)
        cv2.putText(frame, f"{name}:{conf_l:.2f}", (x1_l, y1_l - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if tl_state is None or conf_l > tl_state[1]:
            tl_state = (name, conf_l)

    light_label = tl_state[0] if tl_state else "no-light"
    cv2.putText(frame, f"Light: {light_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if light_label == 'green' else (0, 0, 255), 2)

    # 3) Xử lý từng xe
    for box in result.boxes:
        tid = int(box.id.cpu().item())
        x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int)[0]
        cx = (x1 + x2) // 2
        cy = y2

        # Khởi tạo history nếu track mới
        if tid not in track_history:
            track_history[tid] = {
                'pt': (cx, cy),
                'crossed': False,
                'violation': False,
                'violation_time': None
            }
        rec = track_history[tid]

        # Màu box
        box_color = (0, 0, 255) if rec['violation'] else (255, 0, 0)

        # Kiểm tra vượt vạch lần đầu
        if not rec['crossed']:
            s_prev = side_of_line(rec['pt'], line_pts[0], line_pts[1])
            s_curr = side_of_line((cx, cy), line_pts[0], line_pts[1])
            if s_prev * s_curr < 0:
                if light_label == 'red':
                    rec['violation'] = True
                    rec['violation_time'] = time.time()
                    crop = result.orig_img[y1:y2, x1:x2]
                    fname = os.path.join('vi_pham', f"car_{tid}_{frame_count}.jpg")
                    cv2.imwrite(fname, crop)
                    log_violation(tid, fname)
                    print(f"[VI PHAM] Saved and logged {fname}")
                rec['crossed'] = True

        # Cập nhật centroid
        rec['pt'] = (cx, cy)

        # Vẽ box và thông tin
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        cv2.circle(frame, (cx, cy), 4, box_color, -1)

        # Hiển thị thông báo vi phạm trong 1s
        if rec['violation'] and rec['violation_time'] is not None:
            if time.time() - rec['violation_time'] <= 1.0:
                cv2.putText(frame, "VI PHAM", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 4) Hiển thị tổng số vi phạm
    violation_count = sum(1 for v in track_history.values() if v['violation'])
    cv2.putText(frame, f"Violations: {violation_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Hiển thị kết quả
    cv2.imshow("Tracking & Violation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
