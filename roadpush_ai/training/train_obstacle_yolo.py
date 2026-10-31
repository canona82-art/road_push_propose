import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from ultralytics import YOLO

# 1. 0617_01.xml의 실제 라벨명을 RoadPush 5대 핵심 클래스로 완벽 매핑
BBOX_LABEL_MAP = {
    # Class 0: PM / 자전거 / 이륜차 (인도 위 방치 킥보드, 자전거, 오토바이)
    "scooter": 0,
    "bicycle": 0,
    "motorcycle": 0,

    # Class 1: 불법 주정차 차량
    "car": 1,
    "truck": 1,
    "bus": 1,

    # Class 2: 공사시설 및 고정 기둥 (바리케이드, 볼라드, 소화전, 전신주/가로수)
    "barricade": 2,
    "bollard": 2,
    "fire_hydrant": 2,
    "pole": 2,
    "tree_trunk": 2,

    # Class 3: 상가 적치물 및 노상 방해물 (입간판, 화분, 의자, 테이블, 벤치, 캐리어 등)
    "movable_signage": 3,
    "potted_plant": 3,
    "chair": 3,
    "table": 3,
    "bench": 3,
    "carrier": 3,
    "kiosk": 3,

    # Class 4: 보행자 및 보행약자 (개인정보 비식별화 모자이크 & 보행 궤적용)
    "person": 4,
    "wheelchair": 4,
    "stroller": 4
}

CLASS_NAMES = {
    0: "PM_Scooter_Bike",
    1: "Vehicle",
    2: "Bollard_Barricade_Pole",
    3: "Signage_Obstacle",
    4: "Person"
}


def prepare_bbox_dataset(raw_bbox_root, output_dir, val_ratio=0.15):
    raw_root = Path(raw_bbox_root).resolve()
    out_root = Path(output_dir).resolve()

    xml_files = sorted(list(raw_root.rglob("*.xml")))
    print(f"[1/3] 발견된 XML 파일 수: {len(xml_files)}개")

    samples = []
    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_folder = xml_path.parent

        for img_tag in root.findall("image"):
            img_name = img_tag.attrib.get("name")
            img_w = float(img_tag.attrib.get("width", 1920))
            img_h = float(img_tag.attrib.get("height", 1080))

            img_file = img_folder / img_name
            if not img_file.exists():
                continue

            yolo_lines = []
            for box in img_tag.findall("box"):
                label = box.attrib.get("label", "").strip()
                if label not in BBOX_LABEL_MAP:
                    continue

                cls_id = BBOX_LABEL_MAP[label]
                xtl = float(box.attrib["xtl"])
                ytl = float(box.attrib["ytl"])
                xbr = float(box.attrib["xbr"])
                ybr = float(box.attrib["ybr"])

                # YOLO 정규화 좌표 (cx, cy, w, h) 계산 및 클리핑
                cx = min(max(((xtl + xbr) / 2.0) / img_w, 0.0), 1.0)
                cy = min(max(((ytl + ybr) / 2.0) / img_h, 0.0), 1.0)
                bw = min(max((xbr - xtl) / img_w, 0.0), 1.0)
                bh = min(max((ybr - ytl) / img_h, 0.0), 1.0)

                if bw > 0.003 and bh > 0.003:
                    yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if yolo_lines:
                samples.append((img_file, yolo_lines))

    print(f"[2/3] 유효 학습 이미지 총 {len(samples)}장 추출 완료! 데이터셋 구성 중...")
    random.seed(42)
    random.shuffle(samples)

    split_idx = int(len(samples) * (1.0 - val_ratio))
    splits = {"train": samples[:split_idx], "val": samples[split_idx:]}

    for split_name, items in splits.items():
        img_out_dir = out_root / "images" / split_name
        lbl_out_dir = out_root / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for idx, (src_img, lines) in enumerate(items):
            # 폴더명_파일명으로 고유 이름 생성 (심볼릭 링크로 디스크 용량 절약!)
            unique_stem = f"{src_img.parent.name}_{src_img.stem}"
            dst_img = img_out_dir / f"{unique_stem}{src_img.suffix}"
            dst_lbl = lbl_out_dir / f"{unique_stem}.txt"

            if not dst_img.exists():
                os.symlink(src_img, dst_img)

            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    yaml_path = out_root / "roadpush_obstacle.yaml"
    yaml_text = f"path: {out_root}\ntrain: images/train\nval: images/val\nnames:\n"
    for k, v in CLASS_NAMES.items():
        yaml_text += f"  {k}: {v}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print(f"[3/3] YOLO 데이터셋 준비 완료: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    # 주한님의 실제 폴더 경로 반영
    RAW_BBOX_DIR = "/home/a/projects/jhs/15.인도보행영상/바운딩박스"
    DATASET_OUT_DIR = "/home/a/projects/jhs/datasets/obstacle_det"

    yaml_file = prepare_bbox_dataset(RAW_BBOX_DIR, DATASET_OUT_DIR)

    # RTX 5090으로 즉시 YOLOv8m 파인튜닝 시작!
    print("=== RTX 5090 YOLOv8m 장애물 탐지 모델 학습 시작 ===")
    model = YOLO("yolov8m.pt")
    model.train(
        data=str(yaml_file),
        epochs=40,
        imgsz=640,
        batch=32,      # RTX 5090 기준 초고속 배치
        device=0,
        workers=8,
        project="/home/a/projects/jhs/roadpush_weights",
        name="obstacle_det_kr"
    )