from __future__ import annotations

import csv
import html
import math
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any


EARTH_RADIUS_METERS = 6_371_000.0


def _match_previous_observations(
    current: list[dict[str, Any]],
    previous: list[dict[str, Any]],
    threshold_meters: float,
) -> dict[int, tuple[dict[str, Any], float]]:
    """Match each current point to the newest prior inspection within the radius."""
    if not current or not previous or threshold_meters <= 0:
        return {}
    all_items = current + previous
    reference_latitude = sum(float(item["latitude"]) for item in all_items) / len(all_items)
    longitude_scale = EARTH_RADIUS_METERS * math.cos(math.radians(reference_latitude))

    def projected(item: dict[str, Any]) -> tuple[float, float]:
        return (
            math.radians(float(item["longitude"])) * longitude_scale,
            math.radians(float(item["latitude"])) * EARTH_RADIUS_METERS,
        )

    video_rank: dict[int, int] = {}
    cells: dict[tuple[int, int], list[tuple[int, dict[str, Any], float, float]]] = {}
    for item in previous:
        source_video_id = int(item["video_id"])
        rank = video_rank.setdefault(source_video_id, len(video_rank))
        x, y = projected(item)
        cell = (math.floor(x / threshold_meters), math.floor(y / threshold_meters))
        cells.setdefault(cell, []).append((rank, item, x, y))

    matches: dict[int, tuple[dict[str, Any], float]] = {}
    for item in current:
        x, y = projected(item)
        cell_x, cell_y = math.floor(x / threshold_meters), math.floor(y / threshold_meters)
        candidates: list[tuple[int, float, dict[str, Any]]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for rank, prior, prior_x, prior_y in cells.get((cell_x + dx, cell_y + dy), ()):
                    distance = math.hypot(x - prior_x, y - prior_y)
                    if distance <= threshold_meters:
                        candidates.append((rank, distance, prior))
        if candidates:
            _rank, distance, prior = min(candidates, key=lambda value: (value[0], value[1]))
            matches[int(item["id"])] = (prior, distance)
    return matches


def _copy_frame(source: str, destination: Path) -> bool:
    path = Path(source)
    if not path.is_file():
        return False
    shutil.copy2(path, destination)
    return True


def _change_label(delta: int) -> str:
    if delta < 0:
        return "개선"
    if delta > 0:
        return "악화"
    return "동일"


def create_inspection_report(repository: Any, data_root: str | Path, video_id: int) -> Path:
    video = repository.get_video(video_id)
    if not video:
        raise ValueError("영상을 찾을 수 없습니다.")
    if video["status"] != "completed":
        raise ValueError("분석이 완료된 영상만 리포트를 만들 수 있습니다.")

    segments = repository.list_segments(video_id)
    observations = repository.list_observations(video_id)
    previous = (
        repository.list_previous_observations(video_id)
        if hasattr(repository, "list_previous_observations")
        else []
    )
    try:
        overlap_meters = max(0.0, float(os.getenv("ROADPUSH_GPS_OVERLAP_METERS", "5")))
    except ValueError:
        overlap_meters = 5.0
    matches = _match_previous_observations(observations, previous, overlap_meters)

    report_directory = Path(data_root) / "reports" / f"video_{video_id}"
    if report_directory.exists():
        shutil.rmtree(report_directory)
    frames_directory = report_directory / "frames"
    frames_directory.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    for segment in segments:
        segment_items = [item for item in observations if item["segment_id"] == segment["id"]]
        compared_items = [item for item in segment_items if int(item["id"]) in matches]
        risky_items = [item for item in segment_items if item["level"] != "safe"]
        if compared_items:
            representative = max(
                compared_items,
                key=lambda item: (
                    abs(int(item["risk_score"]) - int(matches[int(item["id"])][0]["risk_score"])),
                    max(int(item["risk_score"]), int(matches[int(item["id"])][0]["risk_score"])),
                ),
            )
        elif risky_items:
            representative = max(risky_items, key=lambda item: item["risk_score"])
        else:
            continue

        extension = Path(representative["frame_path"]).suffix or ".jpg"
        current_image = f"segment_{segment['segment_index']:03d}_after{extension}"
        _copy_frame(representative["frame_path"], frames_directory / current_image)
        prior_match = matches.get(int(representative["id"]))
        prior = prior_match[0] if prior_match else None
        distance = prior_match[1] if prior_match else None
        previous_image = None
        if prior:
            prior_extension = Path(prior["frame_path"]).suffix or ".jpg"
            previous_image = (
                f"segment_{segment['segment_index']:03d}_before_video_{prior['video_id']}"
                f"{prior_extension}"
            )
            if not _copy_frame(prior["frame_path"], frames_directory / previous_image):
                previous_image = None
        rows.append({
            "segment": segment,
            "observation": representative,
            "image": current_image,
            "previous": prior,
            "previous_image": previous_image,
            "match_distance": distance,
        })

    csv_path = report_directory / "inspection_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "100m 구간", "시작 거리(m)", "종료 거리(m)", "구간 거리(m)",
            "시작 시각(초)", "종료 시각(초)", "현재 평균 위험도", "현재 최대 위험도",
            "현재 안전 점수", "현재 등급", "대표 위도", "대표 경도", "현재 프레임",
            "이전 영상", "이전 촬영 시각", "이전 위험도", "이전 판정", "이전 프레임",
            "위험도 변화", "변화 판정", "GPS 매칭 거리(m)",
        ])
        for row in rows:
            segment, observation, prior = row["segment"], row["observation"], row["previous"]
            delta = int(observation["risk_score"]) - int(prior["risk_score"]) if prior else None
            writer.writerow([
                segment["segment_index"] + 1,
                f"{segment['start_distance_meters']:.1f}",
                f"{segment['end_distance_meters']:.1f}",
                f"{segment['distance_meters']:.1f}",
                f"{segment['start_seconds']:.1f}",
                f"{segment['end_seconds']:.1f}",
                f"{segment['average_risk']:.1f}",
                segment["max_risk"],
                f"{segment['safety_score']:.1f}",
                segment["level"],
                f"{observation['latitude']:.6f}",
                f"{observation['longitude']:.6f}",
                f"frames/{row['image']}",
                prior.get("video_name", prior["video_id"]) if prior else "",
                (prior.get("captured_at") or "") if prior else "",
                prior["risk_score"] if prior else "",
                prior["grade"] if prior else "",
                f"frames/{row['previous_image']}" if row["previous_image"] else "",
                delta if delta is not None else "",
                _change_label(delta) if delta is not None else "신규",
                f"{row['match_distance']:.2f}" if row["match_distance"] is not None else "",
            ])

    cards: list[str] = []
    comparison_count = 0
    for row in rows:
        segment, observation, prior = row["segment"], row["observation"], row["previous"]
        current_figure = (
            f'<figure><figcaption>이후 · {html.escape(str(video["original_name"]))}</figcaption>'
            f'<img src="frames/{html.escape(row["image"])}" alt="현재 분석 프레임"></figure>'
        )
        comparison_html = '<p class="new-location">이 위치와 겹치는 이전 점검 데이터가 없습니다.</p>'
        if prior:
            comparison_count += 1
            delta = int(observation["risk_score"]) - int(prior["risk_score"])
            prior_image = (
                f'<img src="frames/{html.escape(row["previous_image"])}" alt="이전 분석 프레임">'
                if row["previous_image"] else '<div class="missing-image">이전 이미지 없음</div>'
            )
            comparison_html = f"""
              <div class="comparison">
                <figure><figcaption>이전 · {html.escape(str(prior.get('video_name', prior['video_id'])))}</figcaption>{prior_image}</figure>
                {current_figure}
              </div>
              <p class="change {_change_label(delta)}"><b>{_change_label(delta)}</b> · 위험도 {prior['risk_score']} → {observation['risk_score']} ({delta:+d}) · 위치 차이 {row['match_distance']:.2f}m</p>
            """
            current_figure = ""
        cards.append(
            f"""
            <article>
              <h2>{segment['segment_index'] + 1}구간 · {html.escape(str(segment['level']).upper())}</h2>
              {comparison_html}{current_figure}
              <dl>
                <div><dt>거리</dt><dd>{segment['start_distance_meters']:.1f}–{segment['end_distance_meters']:.1f}m</dd></div>
                <div><dt>영상 시각</dt><dd>{segment['start_seconds']:.1f}–{segment['end_seconds']:.1f}초</dd></div>
                <div><dt>현재 위험도</dt><dd>평균 {segment['average_risk']:.1f} / 최대 {segment['max_risk']}</dd></div>
                <div><dt>GPS</dt><dd>{observation['latitude']:.6f}, {observation['longitude']:.6f}</dd></div>
                <div><dt>현재 판정</dt><dd>{html.escape(str(observation['grade']))}</dd></div>
              </dl>
            </article>
            """
        )

    safety_score = float(video.get("safety_score") or 0)
    html_path = report_directory / "inspection_report.html"
    html_path.write_text(
        f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>RoadPush 점검 리포트</title>
        <style>body{{font-family:sans-serif;max-width:1100px;margin:40px auto;color:#172033}}header{{border-bottom:3px solid #1687a7;margin-bottom:24px}}.summary{{display:flex;gap:30px;flex-wrap:wrap}}article{{break-inside:avoid;border:1px solid #ccd5df;border-radius:12px;padding:18px;margin:18px 0}}.comparison{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}figure{{margin:10px 0}}figcaption{{font-weight:700;margin-bottom:7px}}img,.missing-image{{width:100%;aspect-ratio:16/9;object-fit:contain;border-radius:8px;background:#eef2f6}}.missing-image{{display:grid;place-items:center;color:#667085}}.change{{padding:10px;border-radius:8px;background:#eef6f8}}.change.개선{{color:#087a61}}.change.악화{{color:#c13644}}.new-location{{color:#667085}}dl div{{display:flex;gap:20px;padding:5px 0}}dt{{width:110px;color:#667085}}dd{{margin:0}}@media(max-width:700px){{.comparison{{grid-template-columns:1fr}}}}</style></head>
        <body><header><h1>RoadPush 보행로 점검 리포트</h1><p>{html.escape(str(video['original_name']))}</p></header>
        <section class="summary"><p><b>총 거리</b> {float(video.get('distance_meters') or 0):.1f}m</p><p><b>종합 안전 점수</b> {safety_score:.1f}점</p><p><b>보고 구간</b> {len(rows)}개</p><p><b>이전·이후 비교</b> {comparison_count}개</p></section>
        {''.join(cards) if cards else '<p>비교 대상 또는 조심·위험 구간이 없습니다.</p>'}</body></html>""",
        encoding="utf-8",
    )

    zip_path = report_directory.with_suffix(".zip")
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in report_directory.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(report_directory))
    repository.update_video(video_id, report_path=str(zip_path))
    return zip_path
