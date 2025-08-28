import os
import cv2
import math
import time
import torch
import numpy as np
from ultralytics import YOLO


# Konfigürasyon
MODEL_PATH = "./yolov8s.pt"   # Model dosyası
VIDEO_PATH = "tehdit.mkv"     # İşlenecek input video
OUTPUT_DIR = "outputs"        # Çıktı klasörü
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output_stabilized_tehtid_v2.mp4") # Çıktı dosyası

# Nesne sınıfları
VESSEL_CLASSES = {"ship", "boat", "fishing_boat"}

# Risk parametreleri
T_HORIZON = 120.0  # TCPA için zaman ufku
D_SAFE = 50.0      # Güvenli mesafe (m) - Çok doğru hesaplanmıyor m cinsinden 
V_MIN = 1e-3
W_TCPA = 0.5
W_DCPA = 0.3
W_HEADING = 0.2
RISK_THRESHOLD_HIGH = 75.0
RISK_THRESHOLD_MED = 50.0

# EMA yumuşatma
EMA_TARGET_VEL_ALPHA = 0.7
EMA_OWN_VEL_ALPHA = 0.65
EMA_RISK_ALPHA = 0.6

# Optik akış ayarları
OPTICAL_FLOW_ROI = [450, 700, 300, 980] 
FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7)

# Model / takip ayarları
CONF_THRES = 0.25
IOU_THRES = 0.50
IMG_SIZE = 640
TRACKER_CFG = "bytetrack.yaml"
FPS_FALLBACK = 30.0

cv2.setUseOptimized(True)
cv2.setNumThreads(0)

# HOMOGRAFİ
SRC_POINTS = np.float32([[100, 700], [1180, 700], [800, 450], [480, 450]])
DST_POINTS = np.float32([[-20, 30], [20, 30], [40, 180], [-40, 180]])

# Fonksiyonlar
def px_to_m_homography(x_px, y_px, M):
    pixel_coords = np.float32([[[x_px, y_px]]])
    meter_coords = cv2.perspectiveTransform(pixel_coords, M)
    return (meter_coords[0][0][0], meter_coords[0][0][1]) if meter_coords is not None else (None, None)

def estimate_ego_velocity(frame_gray, prev_gray, M, fps):
    mask = np.zeros_like(frame_gray)
    y1, y2, x1, x2 = OPTICAL_FLOW_ROI
    mask[y1:y2, x1:x2] = 255

    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **FEATURE_PARAMS)
    if p0 is None or len(p0) < 10:
        return (0.0, 0.0)

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]
    if len(good_new) < 5:
        return (0.0, 0.0)

    flow_vectors = good_new - good_old
    median_flow_px = np.median(flow_vectors, axis=0)

    origin_m = px_to_m_homography(0, 0, M)
    flow_end_m = px_to_m_homography(median_flow_px[0], median_flow_px[1], M)
    if origin_m[0] is None or flow_end_m[0] is None:
        return (0.0, 0.0)

    flow_m_x = flow_end_m[0] - origin_m[0]
    flow_m_y = flow_end_m[1] - origin_m[1]
    vx_bg = flow_m_x * fps
    vy_bg = flow_m_y * fps
    return (-vx_bg, -vy_bg)

def update_track(tracks, obj_id, cls_name, bbox, frame_idx, fps, M):
    x_px = (bbox[0] + bbox[2]) / 2
    y_px = bbox[3]

    x_m, y_m = px_to_m_homography(x_px, y_px, M)
    if x_m is None:
        return None

    width_px = bbox[2] - bbox[0]
    st = tracks.get(obj_id)
    if st is None:
        st = {
            "x_m": x_m, "y_m": y_m, "vx": 0.0, "vy": 0.0,
            "last_frame": frame_idx, "cls": cls_name, "risk": 0.0,
            "width_px": width_px, "distance": y_m
        }
        tracks[obj_id] = st
        return st

    dt = max((frame_idx - st["last_frame"]) / float(fps), 1.0/fps)
    vx_i = (x_m - st["x_m"]) / dt
    vy_i = (y_m - st["y_m"]) / dt

    st["vx"] = EMA_TARGET_VEL_ALPHA * vx_i + (1.0 - EMA_TARGET_VEL_ALPHA) * st["vx"]
    st["vy"] = EMA_TARGET_VEL_ALPHA * vy_i + (1.0 - EMA_TARGET_VEL_ALPHA) * st["vy"]

    st["x_m"] = x_m
    st["y_m"] = y_m
    st["last_frame"] = frame_idx
    st["cls"] = cls_name
    st["width_px"] = width_px
    st["distance"] = y_m

    return st

def calculate_risk(p_own, v_own, p_target, v_target):
    r = np.array([p_target[0] - p_own[0], p_target[1] - p_own[1]], dtype=float)
    v = np.array([v_target[0] - v_own[0], v_target[1] - v_own[1]], dtype=float)

    v2 = np.dot(v, v)
    if v2 < V_MIN:
        tcpa = float("inf")
        dcpa = np.linalg.norm(r)
    else:
        tcpa = -np.dot(r, v) / v2
        dcpa = np.linalg.norm(r + v * tcpa)

    r_tcpa = 1.0 - (tcpa / T_HORIZON) if 0 <= tcpa <= T_HORIZON else 0.0
    r_dcpa = 1.0 - (dcpa / D_SAFE) if dcpa <= D_SAFE else 0.0
    dot_product = np.dot(r, v)
    r_heading = 1.0 if dot_product < 0 else 0.0

    risk_score = (W_TCPA * r_tcpa + W_DCPA * r_dcpa) * (1 + W_HEADING * r_heading)
    risk_score = float(np.clip(100.0 * risk_score, 0.0, 100.0))

    return risk_score, tcpa, dcpa

def color_for_risk(score):
    if score >= RISK_THRESHOLD_HIGH: return (0, 0, 255)
    elif score >= RISK_THRESHOLD_MED: return (0, 255, 255)
    else: return (0, 255, 0)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        HOMOGRAPHY_MATRIX = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
        print("Homografi matrisi oluşturuldu.")
    except Exception as e:
        print(f"HATA: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {VIDEO_PATH}")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), in_fps, (width, height))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH)
    model.to(device)
    print(f"Model '{MODEL_PATH}' {device} üzerinde.")
    class_names = model.names

    tracks = {}
    frame_idx = 0
    t0 = time.time()
    v_own_smoothed = np.array([0.0, 0.0])
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            v_own_raw = estimate_ego_velocity(frame_gray, prev_gray, HOMOGRAPHY_MATRIX, in_fps)
            v_own_smoothed = EMA_OWN_VEL_ALPHA * np.array(v_own_raw) + (1.0 - EMA_OWN_VEL_ALPHA) * v_own_smoothed
        prev_gray = frame_gray.copy()
        v_own = tuple(v_own_smoothed)

        results = model.track(source=frame, conf=CONF_THRES, iou=IOU_THRES, imgsz=IMG_SIZE,
                              tracker=TRACKER_CFG, persist=True, verbose=False, device=device)

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            for box in results[0].boxes:
                obj_id = int(box.id.item())
                cls_id = int(box.cls.item())
                cls_name = class_names[cls_id]

                if cls_name not in VESSEL_CLASSES:
                    continue

                bbox = box.xyxy.int().cpu().numpy()[0]
                st = update_track(tracks, obj_id, cls_name, bbox, frame_idx, in_fps, HOMOGRAPHY_MATRIX)

                if st:
                    p_own = (0.0, 0.0)
                    p_target = (st["x_m"], st["y_m"])
                    v_target = (st["vx"], st["vy"])

                    risk_raw, tcpa, dcpa = calculate_risk(p_own, v_own, p_target, v_target)

                    st["risk"] = EMA_RISK_ALPHA * risk_raw + (1 - EMA_RISK_ALPHA) * st["risk"]
                    risk = st["risk"]

                    color = color_for_risk(risk)
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    tcpa_txt = f"{tcpa:.1f}s" if math.isfinite(tcpa) and tcpa >= 0 else "N/A"
                    label = f"ID:{obj_id} R:{risk:.1f} D:{st['distance']:.1f}m TCPA:{tcpa_txt} DCPA:{dcpa:.1f}m"
                    cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

        v_own_kmh = math.sqrt(v_own[0]**2 + v_own[1]**2) * 3.6
        cv2.putText(frame, f"Ego Hizi: {v_own_kmh:.1f} km/h", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        y1, y2, x1, x2 = OPTICAL_FLOW_ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

        writer.write(frame)
        frame_idx += 1
        if frame_idx > 0 and frame_idx % 120 == 0:
            elapsed = time.time() - t0
            print(f"[{frame_idx}/{total_frames}] ~{frame_idx / max(elapsed, 1e-6):.1f} FPS")

    cap.release()
    writer.release()
    print(f"\nİşlem tamamlandı: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
