import os
import cv2
import numpy as np
from ultralytics import YOLO

class RoadPushAnalyzer:
    def __init__(self,
                 seg_weight="roadpush_weights/sidewalk_seg_kr/weights/best.pt",
                 det_weight="roadpush_weights/obstacle_det_kr/weights/best.pt"):
        self.seg_path = self._find_latest_weight(seg_weight, "sidewalk_seg_kr", "yolov8m-seg.pt")
        self.det_path = self._find_latest_weight(det_weight, "obstacle_det_kr", "yolov8m.pt")
        print(f"[RoadPush] Seg Model: {self.seg_path}")
        print(f"[RoadPush] Det Model: {self.det_path}")

        self.seg_model = YOLO(self.seg_path)
        self.det_model = YOLO(self.det_path)

        self.obstacle_classes = {
            0: "PM/Scooter/Bike",
            1: "Vehicle",
            2: "Bollard/Pole",
            3: "Signage/Obstacle"
        }
        self.person_class_id = 4

        self.recent_eff_ratios = []
        self.recent_penalties = []
        self.disconnect_streak = 0  # ★ 초반 1~2프레임 스파이크(튀는 현상) 방지용 카운터
        self.last_valid_r_eff = 1.0

    @staticmethod
    def _find_latest_weight(default_path, prefix, fallback):
        if os.path.exists(default_path):
            return default_path
        base_dir = "roadpush_weights"
        if os.path.exists(base_dir):
            candidates = sorted(
                [d for d in os.listdir(base_dir) if d.startswith(prefix)],
                reverse=True
            )
            for c in candidates:
                cand_pt = os.path.join(base_dir, c, "weights", "best.pt")
                if os.path.exists(cand_pt):
                    return cand_pt
        return fallback

    def reset_metrics(self):
        self.recent_eff_ratios = []
        self.recent_penalties = []
        self.disconnect_streak = 0
        self.last_valid_r_eff = 1.0

    @staticmethod
    def _reduce_shadow_contrast(frame_bgr):
        """
        ★ 핵심 개선 1 (그림자 경계 완화):
        강한 햇빛과 담벼락 그림자의 극단적인 명암 차이를 LAB 색공간 CLAHE로 부드럽게 평탄화하여,
        세그멘테이션 모델이 그림자 안쪽과 바깥쪽(햇빛 구역)을 하나의 보도로 인식하게 만듦!
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # 어두운 그림자 영역의 밝기를 끌어올리고 과노출된 햇빛 영역과의 편차를 줄임
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        merged = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def get_masks(self, frame_bgr):
        """
        ★ 그림자 절단 & 점자블록/보도블록 마스크 깨짐 완벽 해결 버전:
        1) 그림자 완화 프레임으로 추론하여 그림자 경계선에서 마스크가 잘리는 현상 해결
        2) 노란 점자블록(Hue 15~34)을 수풀로 오인해 구멍 내던 버그 수정 (Hue >= 38만 수풀로 간주)
        3) 마스크 내부에 숭숭 뚫린 구멍(Hole)들을 컨투어 내부 채움으로 매끄럽게 복원
        4) 햇빛 받은 전방 보도 폴리곤이 삭제되지 않도록 하단 연결 기준을 0.65*h -> 0.45*h로 완화
        """
        h, w, _ = frame_bgr.shape

        # 그림자 명암차를 완화한 이미지로 세그멘테이션 추론 수행
        norm_frame = self._reduce_shadow_contrast(frame_bgr)
        results = self.seg_model.predict(norm_frame, imgsz=960, conf=0.16, iou=0.5, retina_masks=True, verbose=False)[0]

        sidewalk_mask = np.zeros((h, w), dtype=np.uint8)
        road_mask = np.zeros((h, w), dtype=np.uint8)
        caution_mask = np.zeros((h, w), dtype=np.uint8)

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)

            for m, cls_id in zip(masks, classes):
                m_bin = (cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0.35).astype(np.uint8)
                if cls_id in [0, 2]:      # 보도(0) + 자전거도로(2)
                    sidewalk_mask |= m_bin
                elif cls_id == 1:         # 차도/골목길(1)
                    road_mask |= m_bin
                elif cls_id == 3:         # 화단/주의구역(3)
                    caution_mask |= m_bin

        # 1) 상단 담벼락/하늘(y < 0.35*h) 원천 차단
        sidewalk_mask[:int(h * 0.35), :] = 0

        # 2) ★ 수정: 노란 점자블록(Hue 15~35)이 파이지 않도록 순수 초록색 수풀(Hue 38~90, 높은 채도)만 제거!
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        green_plant = (cv2.inRange(hsv, (38, 55, 25), (90, 255, 255)) > 0).astype(np.uint8)

        # 차도 방어벽 (확실한 차도 영역 침범 방지)
        road_guard = cv2.dilate(road_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        walkable_mask = sidewalk_mask & (~road_guard) & (~green_plant)

        # 3) ★ 핵심: 보도블록 줄눈이나 점자블록 때문에 그물처럼 깨진 마스크를 매끈하게 하나로 봉합!
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
        walkable_mask = cv2.morphologyEx(walkable_mask, cv2.MORPH_CLOSE, close_kernel)

        # 유효 보도 폴리곤 필터링 (햇빛 받은 우측 전방 보도도 살아남도록 기준을 0.45*h로 완화)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(walkable_mask, connectivity=8)
        filtered_walkable = np.zeros((h, w), dtype=np.uint8)
        min_blob_area = h * w * 0.015

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            blob_y_bottom = stats[label_id, cv2.CC_STAT_TOP] + stats[label_id, cv2.CC_STAT_HEIGHT]
            if area >= min_blob_area and blob_y_bottom >= int(h * 0.45):
                filtered_walkable[labels == label_id] = 1

        # ★ 보도 마스크 안쪽에 뚫려 있는 빈 구멍(점자블록/그림자 경계 구멍)들을 완벽히 메움!
        ext_contours, _ = cv2.findContours(filtered_walkable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if ext_contours:
            cv2.drawContours(filtered_walkable, ext_contours, -1, 1, thickness=cv2.FILLED)
            filtered_walkable = filtered_walkable & (~road_guard) & (~green_plant)

        if np.sum(filtered_walkable) < (h * w * 0.025):
            filtered_walkable = np.zeros((h, w), dtype=np.uint8)

        hull_mask = filtered_walkable.copy()
        if ext_contours:
            all_pts = np.vstack(ext_contours)
            hull = cv2.convexHull(all_pts)
            cv2.drawContours(hull_mask, [hull], -1, 1, thickness=cv2.FILLED)
            hull_mask = hull_mask & (~road_guard) & (~green_plant)

        return filtered_walkable, hull_mask, road_mask, caution_mask

    def detect_and_filter_obstacles(self, frame_bgr, walkable_mask, hull_mask, road_mask, caution_mask, enable_blur=True):
        h, w, _ = frame_bgr.shape
        results = self.det_model.predict(frame_bgr, conf=0.25, iou=0.45, verbose=False)[0]
        valid_obstacles = []

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.08), int(h * 0.05)))
        expanded_sw = cv2.dilate(walkable_mask | hull_mask, dilate_kernel)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)

            if cls_id == self.person_class_id and enable_blur:
                face_y2 = y1 + max(10, int(box_h * 0.28))
                roi = frame_bgr[y1:min(h, face_y2), x1:x2]
                if roi.size > 0:
                    frame_bgr[y1:min(h, face_y2), x1:x2] = cv2.GaussianBlur(roi, (31, 31), 0)

            elif cls_id in self.obstacle_classes:
                if y2 < int(h * 0.38) or (box_w * box_h) < (h * w * 0.002):
                    continue

                bottom_y1 = max(0, y2 - int(box_h * 0.25))
                bottom_y2 = min(h, y2 + 12)

                sw_patch = expanded_sw[bottom_y1:bottom_y2, x1:x2]
                road_patch = road_mask[bottom_y1:bottom_y2, x1:x2]
                caution_patch = caution_mask[bottom_y1:bottom_y2, x1:x2]

                if sw_patch.size > 0:
                    sw_overlap = np.sum(sw_patch > 0) / float(sw_patch.size)
                    road_overlap = np.sum(road_patch > 0) / float(road_patch.size)
                    caution_overlap = np.sum(caution_patch > 0) / float(caution_patch.size)

                    if cls_id == 1 and (road_overlap > 0.25 or sw_overlap < 0.40):
                        continue
                    if cls_id == 2 and (caution_overlap > 0.25 or x1 > int(w * 0.72)):
                        continue

                    if sw_overlap >= 0.10:
                        valid_obstacles.append({
                            "bbox": (x1, y1, x2, y2),
                            "label": self.obstacle_classes[cls_id],
                            "conf": float(box.conf[0]),
                            "cls_id": cls_id
                        })

        return valid_obstacles

    def calculate_effective_width_ratio(self, walkable_mask, hull_mask, obstacles):
        h, w = walkable_mask.shape

        if np.sum(walkable_mask) < (h * w * 0.025):
            return 0.0, None, [], True

        obs_mask = np.zeros_like(walkable_mask)
        obs_scan_ys = []
        for obs in obstacles:
            x1, y1, x2, y2 = obs["bbox"]
            obs_y1 = max(y1, y2 - int((y2 - y1) * 0.70))
            obs_mask[obs_y1:y2, x1:x2] = 1
            target_y = np.clip(y2 - int((y2 - y1) * 0.15), int(h * 0.45), int(h * 0.92))
            obs_scan_ys.append(int(target_y))

        free_walkable_mask = walkable_mask & (~obs_mask)

        base_scan_ys = list(np.linspace(int(h * 0.50), int(h * 0.88), num=6, dtype=int))
        scan_y_list = sorted(list(set(base_scan_ys + obs_scan_ys)))

        line_results = []
        for y in scan_y_list:
            sw_pixels = np.where(walkable_mask[y, :] > 0)[0]
            if len(sw_pixels) < w * 0.05:
                continue

            sw_x1, sw_x2 = int(sw_pixels[0]), int(sw_pixels[-1])
            has_obs_on_line = np.sum(obs_mask[y, :]) > 0

            if has_obs_on_line:
                combined_pixels = np.where((hull_mask[y, :] > 0) | (obs_mask[y, :] > 0) | (walkable_mask[y, :] > 0))[0]
                ideal_x1, ideal_x2 = int(combined_pixels[0]), int(combined_pixels[-1])
            else:
                ideal_x1, ideal_x2 = sw_x1, sw_x2

            total_ideal_width = max(1, ideal_x2 - ideal_x1)

            if not has_obs_on_line:
                ratio = 1.0
                free_x1, free_x2 = ideal_x1, ideal_x2
            else:
                free_row = free_walkable_mask[y, ideal_x1:ideal_x2 + 1]
                padded = np.concatenate(([0], free_row, [0]))
                diffs = np.diff(padded)
                run_starts = np.where(diffs == 1)[0]
                run_ends = np.where(diffs == -1)[0]

                if len(run_starts) > 0:
                    run_lengths = run_ends - run_starts
                    max_idx = np.argmax(run_lengths)
                    max_free_width = int(run_lengths[max_idx])
                    free_x1 = ideal_x1 + int(run_starts[max_idx])
                    free_x2 = ideal_x1 + int(run_ends[max_idx])
                else:
                    max_free_width = 0
                    free_x1, free_x2 = ideal_x1, ideal_x1

                ratio = np.clip(float(max_free_width) / float(total_ideal_width), 0.0, 1.0)

            line_results.append({
                "ratio": ratio, "y": y,
                "sw_x1": ideal_x1, "sw_x2": ideal_x2,
                "free_x1": free_x1, "free_x2": free_x2
            })

        # 스캔라인이 1개라도 잡히면 정상으로 인정하여 초반(00:09, 00:13) 순간 스파이크 방지!
        if len(line_results) == 0:
            return 0.0, None, [], True

        bottleneck = min(line_results, key=lambda x: x["ratio"])
        return bottleneck["ratio"], bottleneck, line_results, False

    def process_frame(self, frame_bgr, enable_blur=True):
        vis_frame = frame_bgr.copy()
        h, w, _ = vis_frame.shape

        walkable_mask, hull_mask, road_mask, caution_mask = self.get_masks(vis_frame)
        obstacles = self.detect_and_filter_obstacles(
            vis_frame, walkable_mask, hull_mask, road_mask, caution_mask, enable_blur=enable_blur
        )
        r_eff, bottleneck, all_scans, raw_disconnected = self.calculate_effective_width_ratio(
            walkable_mask, hull_mask, obstacles
        )

        # ★ 핵심 개선 3: 1~2프레임 일시적 마스크 흔들림으로 인한 초반 위험도 스파이크 완벽 차단!
        # 연속 6프레임 이상 보도가 사라졌을 때만 실제 '보도 단절(DISCONNECTED)'로 판정
        if raw_disconnected:
            self.disconnect_streak += 1
            if self.disconnect_streak < 6:
                is_disconnected = False
                r_eff = self.last_valid_r_eff  # 잠깐 튄 프레임은 직전 정상 유효폭 유지
            else:
                is_disconnected = True
                r_eff = 0.0
        else:
            self.disconnect_streak = 0
            is_disconnected = False
            self.last_valid_r_eff = r_eff

        obs_penalty = 0.0
        for obs in obstacles:
            _, y1, _, y2 = obs["bbox"]
            if y2 > h * 0.45:
                obs_penalty += 0.12 if obs["cls_id"] in [0, 3] else 0.08
        obs_penalty = min(obs_penalty, 0.35)

        self.recent_eff_ratios.append(r_eff)
        self.recent_penalties.append(obs_penalty)
        if len(self.recent_eff_ratios) > 10:
            self.recent_eff_ratios.pop(0)
            self.recent_penalties.pop(0)

        smooth_r_eff = float(np.mean(self.recent_eff_ratios))
        smooth_penalty = float(np.mean(self.recent_penalties))

        if is_disconnected:
            risk_score = int(np.clip(85 + (1.0 - smooth_r_eff) * 15, 85, 100))
        else:
            raw_risk = ((1.0 - smooth_r_eff) * 0.85 + smooth_penalty) * 100.0
            risk_score = int(np.clip(raw_risk, 0, 100))

        overlay = np.zeros_like(vis_frame)
        overlay[walkable_mask == 1] = (0, 220, 80)
        vis_frame = cv2.addWeighted(vis_frame, 1.0, overlay, 0.35, 0)

        for obs in obstacles:
            x1, y1, x2, y2 = obs["bbox"]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(vis_frame, f"{obs['label']} {obs['conf']:.2f}", (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        for scan in all_scans:
            sy = scan["y"]
            cv2.line(vis_frame, (scan["sw_x1"], sy), (scan["sw_x2"], sy), (180, 180, 180), 1)

        if bottleneck is not None:
            by = bottleneck["y"]
            cv2.line(vis_frame, (bottleneck["sw_x1"], by - 18), (bottleneck["sw_x1"], by + 18), (0, 255, 255), 3)
            cv2.line(vis_frame, (bottleneck["sw_x2"], by - 18), (bottleneck["sw_x2"], by + 18), (0, 255, 255), 3)

            fx1, fx2 = bottleneck["free_x1"], bottleneck["free_x2"]
            arrow_color = (255, 220, 0) if r_eff >= 0.65 else ((0, 165, 255) if r_eff >= 0.40 else (0, 0, 255))
            if fx2 - fx1 > 10:
                cv2.arrowedLine(vis_frame, (fx1, by), (fx2, by), arrow_color, 4, tipLength=0.04)
                cv2.arrowedLine(vis_frame, (fx2, by), (fx1, by), arrow_color, 4, tipLength=0.04)
                mid_x = (fx1 + fx2) // 2
                cv2.putText(vis_frame, f"Passable: {int(r_eff * 100)}%", (max(10, mid_x - 85), by - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, arrow_color, 2)
        elif is_disconnected:
            cv2.putText(vis_frame, "WARNING: SIDEWALK DISCONNECTED!", (int(w * 0.08), int(h * 0.72)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        status_text, color = self.get_risk_grade(risk_score, is_disconnected)
        cv2.rectangle(vis_frame, (15, 15), (520, 115), (20, 25, 30), -1)
        cv2.rectangle(vis_frame, (15, 15), (520, 115), color, 3)
        cv2.putText(vis_frame, f"Effective Sidewalk Width: {int(smooth_r_eff * 100)}%",
                    (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Risk Score: {risk_score} ({status_text})",
                    (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        return vis_frame, smooth_r_eff, risk_score, status_text

    @staticmethod
    def get_risk_grade(score, is_disconnected=False):
        if is_disconnected:
            return "DISCONNECTED (DANGER)", (0, 0, 255)
        if score <= 25:
            return "SAFE", (0, 230, 100)
        elif score <= 50:
            return "CAUTION", (0, 215, 255)
        elif score <= 75:
            return "WARNING", (0, 140, 255)
        else:
            return "DANGER / OBSTRUCTED", (0, 0, 255)