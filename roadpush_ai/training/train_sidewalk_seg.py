import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from ultralytics import YOLO

# 15_SM0915_01.xml 실제 라벨 매핑
# Class 0: Sidewalk (보행 가능 인도 = 일반 보도 + 점자블록 + 맨홀/배수로 뚜껑)
# Class 1: Roadway (차도 및 골목길/이면도로)
# Class 2: Bike_Lane (자전거 도로)
# Class 3: Non_Walkable_Zone (가로수 화단 tree_zone, 계단 stairs, 보수공사 구역 repair_zone)

CLASS_NAMES = {
    0: "Sidewalk",
    1: "Roadway",
    2: "Bike_Lane",
    3: "Caution_Tree_Zone"
}

def map_surface_label(poly_tag):
    label = poly_tag.attrib.get("label", "").strip()
    attr_val = ""
    for attr in poly_tag.findall("attribute"):
        if attr.attrib.get("name") == "attribute" and attr.text:
            attr_val = attr.text.strip()

    if label in ["sidewalk", "braille_guide_blocks"]:
        return 0  # 보행 가능 보도
    elif label in ["roadway", "alley"]:
        return 1  # 차도 / 이면도로
    elif label == "bike_lane":
        return 2  # 자전거 도로
    elif label == "caution_zone":
        # 맨홀(manhole)이나 배수로(grating)는 보도 위에 평평하게 있어 보행 가능하므로 0번(Sidewalk) 처리,
        # 가로수 구역(tree_zone), 계단(stairs), 보수 구역(repair_zone)은 보행 방해 구역(3번)으로 분리!
        if attr_val in ["tree_zone", "stairs", "repair_zone"]:
            return 3
        else:
            return 0
    return None


def prepare_surface_seg_dataset(raw_surface_root, output_dir, val_ratio=0.15):
    raw_root = Path(raw_surface_root).resolve()
    out_root = Path(output_dir).resolve()

    xml_files = sorted(list(raw_root.rglob("*.xml")))
    print(f"[1/3] 서피스마스킹 XML 파일 수: {len(xml_files)}개")

    samples = []
    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        xml_dir = xml_path.parent

        for img_tag in root.findall("image"):
            img_name = img_tag.attrib.get("name")
            img_w = float(img_tag.attrib.get("width", 1920))
            img_h = float(img_tag.attrib.get("height", 1080))

            # 원본 JPG 파일 찾기 (XML과 같은 폴더 또는 하위 폴더 검색, mask 폴더 제외)
            img_file = xml_dir / img_name
            if not img_file.exists():
                candidates = [p for p in xml_dir.rglob(img_name) if "mask" not in p.parts]
                if candidates:
                    img_file = candidates[0]
                else:
                    continue

            yolo_seg_lines = []
            for poly in img_tag.findall("polygon"):
                cls_id = map_surface_label(poly)
                if cls_id is None:
                    continue

                pts_str = poly.attrib.get("points", "").split(";")
                norm_coords = []
                for pt in pts_str:
                    if not pt.strip():
                        continue
                    x_str, y_str = pt.split(",")
                    nx = min(max(float(x_str) / img_w, 0.0), 1.0)
                    ny = min(max(float(y_str) / img_h, 0.0), 1.0)
                    norm_coords.append(f"{nx:.6f} {ny:.6f}")

                # 폴리곤은 최소 점 3개 이상이어야 유효
                if len(norm_coords) >= 3:
                    yolo_seg_lines.append(f"{cls_id} " + " ".join(norm_coords))

            if yolo_seg_lines:
                samples.append((img_file, yolo_seg_lines))

    print(f"[2/3] 유효 세그멘테이션 이미지 총 {len(samples)}장 추출 완료!")
    random.seed(42)
    random.shuffle(samples)

    split_idx = int(len(samples) * (1.0 - val_ratio))
    splits = {"train": samples[:split_idx], "val": samples[split_idx:]}

    for split_name, items in splits.items():
        img_out_dir = out_root / "images" / split_name
        lbl_out_dir = out_root / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for src_img, lines in items:
            unique_stem = f"{src_img.parent.name}_{src_img.stem}"
            dst_img = img_out_dir / f"{unique_stem}{src_img.suffix}"
            dst_lbl = lbl_out_dir / f"{unique_stem}.txt"

            if not dst_img.exists():
                os.symlink(src_img, dst_img)

            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    yaml_path = out_root / "roadpush_sidewalk_seg.yaml"
    yaml_text = f"path: {out_root}\ntrain: images/train\nval: images/val\nnames:\n"
    for k, v in CLASS_NAMES.items():
        yaml_text += f"  {k}: {v}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print(f"[3/3] YOLOv8-seg 데이터셋 준비 완료: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    # 서피스마스킹 압축 푼 폴더 경로로 지정해 주세요
    RAW_SURFACE_DIR = "/home/a/projects/jhs/15.인도보행영상/서피스마스킹"
    DATASET_OUT_DIR = "/home/a/projects/jhs/datasets/sidewalk_seg"

    yaml_file = prepare_surface_seg_dataset(RAW_SURFACE_DIR, DATASET_OUT_DIR)

    print("=== RTX 5090 YOLOv8m-seg 보도 분할 모델 학습 시작 ===")
    model = YOLO("yolov8m-seg.pt")
    model.train(
        data=str(yaml_file),
        epochs=40,
        imgsz=640,
        batch=32,
        device=0,
        workers=8,
        project="/home/a/projects/jhs/roadpush_weights",
        name="sidewalk_seg_kr"
    )