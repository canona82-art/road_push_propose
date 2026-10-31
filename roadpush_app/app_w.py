from __future__ import annotations

import io
import hashlib
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from database import MySQLRepository
from reporting import create_inspection_report
from segmentation import SegmentArchive
from video_processing import VideoProcessor


EARTH_RADIUS_METERS = 6_371_000.0


def latest_overlapping_observations(
    observations: list[dict[str, Any]],
    overlap_meters: float = 5.0,
) -> list[dict[str, Any]]:
    """Keep all same-video points while newer videos supersede nearby older points."""
    if not observations or overlap_meters <= 0:
        return list(observations)

    reference_latitude = sum(float(item["latitude"]) for item in observations) / len(observations)
    longitude_scale = EARTH_RADIUS_METERS * math.cos(math.radians(reference_latitude))
    latitude_scale = EARTH_RADIUS_METERS

    def projected(item: dict[str, Any]) -> tuple[float, float]:
        return (
            math.radians(float(item["longitude"])) * longitude_scale,
            math.radians(float(item["latitude"])) * latitude_scale,
        )

    kept: list[dict[str, Any]] = []
    newer_cells: dict[tuple[int, int], list[tuple[float, float]]] = {}
    index = 0
    while index < len(observations):
        video_id = observations[index]["video_id"]
        end = index
        while end < len(observations) and observations[end]["video_id"] == video_id:
            end += 1
        video_coordinates: list[tuple[float, float]] = []
        for item in observations[index:end]:
            x, y = projected(item)
            video_coordinates.append((x, y))
            cell_x, cell_y = math.floor(x / overlap_meters), math.floor(y / overlap_meters)
            overlaps_newer = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    candidates = newer_cells.get((cell_x + dx, cell_y + dy), ())
                    if any(
                        math.hypot(x - other_x, y - other_y) <= overlap_meters
                        for other_x, other_y in candidates
                    ):
                        overlaps_newer = True
                        break
                if overlaps_newer:
                    break
            if not overlaps_newer:
                kept.append(item)

        # Points in the same video do not supersede each other.
        # Hidden points still represent newer coverage for even older inspections.
        for x, y in video_coordinates:
            cell = (math.floor(x / overlap_meters), math.floor(y / overlap_meters))
            newer_cells.setdefault(cell, []).append((x, y))
        index = end
    return kept


@dataclass
class Observation:
    id: int
    video_id: str
    frame_index: int
    latitude: float
    longitude: float
    risk: float
    video_level: str
    captured_at: str
    segment_index: int | None = None
    segment_finalized: bool = False

    @property
    def frame_level(self) -> str:
        if self.risk < 0.35:
            return "safe"
        if self.risk < 0.7:
            return "caution"
        return "danger"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["frame_level"] = self.frame_level
        result["frame_url"] = f"/api/observations/{self.id}/frame"
        result["video_url"] = (
            f"/api/videos/{quote(self.video_id, safe='')}/segments/{self.segment_index}/video"
            if self.segment_index is not None and self.segment_finalized
            else None
        )
        return result


class ObservationStore:
    """Thread-safe in-memory store. Replace with a DB for production use."""

    def __init__(self) -> None:
        self._items: list[Observation] = []
        self._frames: dict[int, tuple[bytes, str]] = {}
        self._lock = threading.Lock()
        self._next_id = 1

    def add(
        self,
        latitude: float,
        longitude: float,
        risk: float,
        video_id: str | None = None,
        frame_index: int | None = None,
        captured_at: str | None = None,
        frame: bytes | None = None,
        frame_mimetype: str = "image/svg+xml",
    ) -> Observation:
        with self._lock:
            item_id = self._next_id
            self._next_id += 1
            resolved_video_id = video_id or f"single-frame-{item_id:06d}"
            video_items = [item for item in self._items if item.video_id == resolved_video_id]
            resolved_frame_index = frame_index if frame_index is not None else len(video_items)
            incoming_level = risk_level(risk)
            current_level = max(
                (item.video_level for item in video_items),
                key=level_rank,
                default="safe",
            )
            video_level = max((current_level, incoming_level), key=level_rank)

            # A later dangerous frame promotes every GPS point of this video.
            if video_level != current_level:
                for previous_item in video_items:
                    previous_item.video_level = video_level

            item = Observation(
                id=item_id,
                video_id=resolved_video_id,
                frame_index=resolved_frame_index,
                latitude=latitude,
                longitude=longitude,
                risk=risk,
                video_level=video_level,
                captured_at=captured_at or datetime.now(timezone.utc).isoformat(),
            )
            self._items.append(item)
            self._frames[item_id] = (
                frame or make_dummy_frame(item),
                frame_mimetype if frame else "image/svg+xml",
            )
            return item

    def list(self, since_id: int = 0, limit: int = 500) -> list[Observation]:
        with self._lock:
            return [item for item in self._items if item.id > since_id][-limit:]

    def frame(self, item_id: int) -> tuple[bytes, str] | None:
        with self._lock:
            return self._frames.get(item_id)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._frames.clear()
            self._next_id = 1

    def video_count(self) -> int:
        with self._lock:
            return len({item.video_id for item in self._items})

    def video_items(self, video_id: str) -> list[Observation]:
        with self._lock:
            return sorted(
                (item for item in self._items if item.video_id == video_id),
                key=lambda item: item.frame_index,
            )


LEVEL_RANK = {"safe": 0, "caution": 1, "danger": 2}


def level_rank(level: str) -> int:
    return LEVEL_RANK[level]


def risk_level(risk: float) -> str:
    if risk < 0.35:
        return "safe"
    if risk < 0.7:
        return "caution"
    return "danger"


def make_dummy_frame(item: Observation) -> bytes:
    colors = {"safe": "#2dd4a8", "caution": "#f4b942", "danger": "#ff5b65"}
    color = colors[item.video_level]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540">
      <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
        <stop stop-color="#142338"/><stop offset="1" stop-color="#07111f"/>
      </linearGradient></defs>
      <rect width="960" height="540" fill="url(#g)"/>
      <path d="M0 500 L380 235 L580 235 L960 500" fill="#202f43"/>
      <path d="M470 540 L485 455 M490 405 L498 360 M505 320 L510 290" stroke="#dce5ef" stroke-width="8"/>
      <circle cx="720" cy="120" r="54" fill="{color}" opacity=".18"/>
      <circle cx="720" cy="120" r="25" fill="{color}"/>
      <text x="48" y="70" fill="#fff" font-family="sans-serif" font-size="30">DUMMY ROAD FRAME #{item.id}</text>
      <text x="48" y="115" fill="#9fb0c3" font-family="sans-serif" font-size="22">GPS {item.latitude:.6f}, {item.longitude:.6f}</text>
      <text x="48" y="153" fill="{color}" font-family="sans-serif" font-size="24" font-weight="bold">FRAME RISK {item.risk:.2f} · VIDEO {item.video_level.upper()}</text>
      <text x="48" y="510" fill="#7f91a6" font-family="sans-serif" font-size="18">Demo frame — replace with the received camera image</text>
    </svg>"""
    return svg.encode("utf-8")


ROAD_ROUTE_TEMPLATES = (
    # Sejong-daero: Seoul City Hall -> Gwanghwamun
    (
        (37.56575, 126.97787),
        (37.56720, 126.97747),
        (37.56935, 126.97699),
        (37.57135, 126.97672),
        (37.57322, 126.97684),
        (37.57505, 126.97686),
    ),
    # Sejong-daero: Seoul City Hall -> Sungnyemun -> Seoul Station
    (
        (37.56575, 126.97787),
        (37.56410, 126.97742),
        (37.56210, 126.97670),
        (37.56005, 126.97530),
        (37.55840, 126.97395),
        (37.55635, 126.97228),
        (37.55472, 126.97095),
    ),
    # Eulji-ro: City Hall junction -> Euljiro 4-ga -> DDP direction
    (
        (37.56605, 126.97825),
        (37.56588, 126.98145),
        (37.56578, 126.98505),
        (37.56575, 126.98905),
        (37.56595, 126.99360),
        (37.56616, 126.99835),
        (37.56605, 127.00305),
        (37.56562, 127.00735),
    ),
)

DUMMY_GPS_SPACING_METERS = 2.0
MIN_POINTS_PER_VIDEO = 120
MAX_POINTS_PER_VIDEO = 200


def interpolate_route(
    route: tuple[tuple[float, float], ...],
    point_count: int,
) -> list[tuple[float, float]]:
    """Resample a road polyline into evenly spaced GPS positions."""
    if point_count <= 1:
        return [route[0]]

    latitude_scale = math.cos(math.radians(sum(point[0] for point in route) / len(route)))
    segment_lengths = []
    for start, end in zip(route, route[1:]):
        delta_lat = end[0] - start[0]
        delta_lng = (end[1] - start[1]) * latitude_scale
        segment_lengths.append(math.hypot(delta_lat, delta_lng))

    total_length = sum(segment_lengths)
    result = []
    segment_index = 0
    distance_before_segment = 0.0

    for point_index in range(point_count):
        target_distance = total_length * point_index / (point_count - 1)
        while (
            segment_index < len(segment_lengths) - 1
            and target_distance > distance_before_segment + segment_lengths[segment_index]
        ):
            distance_before_segment += segment_lengths[segment_index]
            segment_index += 1

        segment_length = segment_lengths[segment_index]
        ratio = (target_distance - distance_before_segment) / segment_length
        start, end = route[segment_index], route[segment_index + 1]
        result.append(
            (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
        )
    return result


def route_length_meters(route: tuple[tuple[float, float], ...]) -> float:
    """Return a sufficiently accurate local distance for short Seoul routes."""
    latitude_scale = math.cos(math.radians(sum(point[0] for point in route) / len(route)))
    return sum(
        math.hypot(
            end[0] - start[0],
            (end[1] - start[1]) * latitude_scale,
        )
        * 111_320
        for start, end in zip(route, route[1:])
    )


def slice_route_by_distance(
    route: tuple[tuple[float, float], ...],
    start_distance_meters: float,
    length_meters: float,
) -> tuple[tuple[float, float], ...]:
    """Return a section of a road, preserving intermediate road vertices."""
    latitude_scale = math.cos(math.radians(sum(point[0] for point in route) / len(route)))
    segment_lengths = []
    for start, end in zip(route, route[1:]):
        segment_lengths.append(
            math.hypot(
                end[0] - start[0],
                (end[1] - start[1]) * latitude_scale,
            )
            * 111_320
        )

    cumulative = [0.0]
    for segment_length in segment_lengths:
        cumulative.append(cumulative[-1] + segment_length)

    end_distance_meters = min(cumulative[-1], start_distance_meters + length_meters)

    def point_at(distance: float) -> tuple[float, float]:
        for index, segment_length in enumerate(segment_lengths):
            if distance <= cumulative[index + 1] or index == len(segment_lengths) - 1:
                ratio = (distance - cumulative[index]) / segment_length
                start, end = route[index], route[index + 1]
                return (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                )
        return route[-1]

    sliced = [point_at(start_distance_meters)]
    for index in range(1, len(route) - 1):
        if start_distance_meters < cumulative[index] < end_distance_meters:
            sliced.append(route[index])
    end_point = point_at(end_distance_meters)
    if end_point != sliced[-1]:
        sliced.append(end_point)
    return tuple(sliced)


def seed_dummy_videos(
    store: ObservationStore,
    video_count: int = 3,
) -> list[str]:
    """Create dense video GPS samples following representative Seoul roads."""
    video_offset = store.video_count()
    generated_video_ids = []

    for video_number in range(video_offset, video_offset + video_count):
        rng = random.Random(20261020 + video_number)
        video_id = f"demo-video-{video_number + 1:03d}"
        generated_video_ids.append(video_id)
        target_level = ("safe", "caution", "danger")[video_number % 3]
        route = ROAD_ROUTE_TEMPLATES[video_number % len(ROAD_ROUTE_TEMPLATES)]
        if rng.random() < 0.5:
            route = tuple(reversed(route))
        point_count = rng.randint(MIN_POINTS_PER_VIDEO, MAX_POINTS_PER_VIDEO)
        section_length = (point_count - 1) * DUMMY_GPS_SPACING_METERS
        available_start_length = route_length_meters(route) - section_length
        start_distance = rng.uniform(0, max(0, available_start_length))
        route = slice_route_by_distance(
            route,
            start_distance_meters=start_distance,
            length_meters=section_length,
        )
        gps_points = interpolate_route(route, point_count)

        risks = [round(rng.uniform(0.05, 0.28), 3) for _ in range(point_count)]
        if target_level == "caution":
            risks[point_count // 2] = 0.53
        elif target_level == "danger":
            risks[point_count // 2] = 0.86

        for frame_index, ((latitude, longitude), risk) in enumerate(zip(gps_points, risks)):
            store.add(
                latitude=latitude,
                longitude=longitude,
                risk=risk,
                video_id=video_id,
                frame_index=frame_index,
            )
    return generated_video_ids


def parse_number(value: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


def archive_complete_video(
    store: ObservationStore,
    archive: SegmentArchive,
    video_id: str,
    fps: float = 10.0,
) -> None:
    archive.reset_video(video_id, fps=fps)
    for item in store.video_items(video_id):
        frame = store.frame(item.id)
        if frame is None:
            continue
        frame_data, frame_mimetype = frame
        archive.add_observation(item, frame_data, frame_mimetype, fps=fps)
    archive.finalize_video(video_id)


def create_legacy_app(
    segment_root: str | Path | None = None,
    encode_segment_videos: bool = False,
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
    store = ObservationStore()
    archive = SegmentArchive(
        root=Path(segment_root) if segment_root else Path(__file__).parent / "data" / "segments",
        encode_videos=encode_segment_videos,
    )
    app.config["SEGMENT_ROOT"] = str(archive.root)
    dummy_video_ids = seed_dummy_videos(store)
    for video_id in dummy_video_ids:
        archive_complete_video(store, archive, video_id)
    app.extensions["observation_store"] = store
    app.extensions["segment_archive"] = archive

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/observations")
    def get_observations():
        try:
            since_id = max(0, int(request.args.get("since_id", 0)))
            limit = min(10_000, max(1, int(request.args.get("limit", 10_000))))
        except ValueError:
            return jsonify({"error": "since_id and limit must be integers"}), 400
        items = store.list(since_id=since_id, limit=limit)
        return jsonify({"observations": [item.to_dict() for item in items]})

    @app.post("/api/observations")
    def create_observation():
        # JSON and multipart/form-data are both accepted for future device integration.
        payload = request.get_json(silent=True) if request.is_json else request.form
        payload = payload or {}
        try:
            latitude = parse_number(payload.get("latitude"), "latitude", -90, 90)
            longitude = parse_number(payload.get("longitude"), "longitude", -180, 180)
            risk = parse_number(payload.get("risk"), "risk", 0, 1)
            fps = parse_number(payload.get("fps", 10), "fps", 0.1, 240)
            frame_index = (
                int(payload["frame_index"])
                if payload.get("frame_index") not in (None, "")
                else None
            )
            if frame_index is not None and frame_index < 0:
                raise ValueError("frame_index must be zero or greater")
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

        uploaded_frame = request.files.get("frame")
        frame = uploaded_frame.read() if uploaded_frame else None
        mimetype = uploaded_frame.mimetype if uploaded_frame else "image/svg+xml"
        item = store.add(
            latitude=latitude,
            longitude=longitude,
            risk=risk,
            video_id=payload.get("video_id"),
            frame_index=frame_index,
            captured_at=payload.get("captured_at"),
            frame=frame,
            frame_mimetype=mimetype,
        )
        stored_frame = store.frame(item.id)
        if stored_frame is not None:
            archive.add_observation(item, *stored_frame, fps=fps)
        if str(payload.get("finalize", "")).lower() in {"1", "true", "yes"}:
            archive.finalize_video(item.video_id)
        return jsonify(item.to_dict()), 201

    @app.get("/api/observations/<int:item_id>/frame")
    def get_frame(item_id: int):
        result = store.frame(item_id)
        if result is None:
            return jsonify({"error": "observation not found"}), 404
        data, mimetype = result
        return Response(io.BytesIO(data), mimetype=mimetype)

    @app.post("/api/dummy/generate")
    def generate_dummy():
        payload = request.get_json(silent=True) or {}
        try:
            count = min(10, max(1, int(payload.get("count", 1))))
        except (TypeError, ValueError):
            return jsonify({"error": "count must be an integer"}), 400
        generated_video_ids = set(seed_dummy_videos(store, video_count=count))
        for video_id in generated_video_ids:
            archive_complete_video(store, archive, video_id)
        items = [item for item in store.list(limit=2_000) if item.video_id in generated_video_ids]
        return jsonify({"observations": [item.to_dict() for item in items]}), 201

    @app.get("/api/videos/<video_id>/segments")
    def get_video_segments(video_id: str):
        return jsonify({"segments": archive.list_segments(video_id)})

    @app.post("/api/videos/<video_id>/finalize")
    def finalize_video(video_id: str):
        segments = archive.finalize_video(video_id)
        if not segments:
            return jsonify({"error": "video not found"}), 404
        return jsonify({"segments": segments})

    @app.get("/api/videos/<video_id>/segments/<int:segment_index>/video")
    def get_segment_video(video_id: str, segment_index: int):
        path = archive.video_path(video_id, segment_index)
        if path is None:
            return jsonify({"error": "encoded segment video not found"}), 404
        return send_file(path, mimetype="video/mp4", conditional=True)

    @app.delete("/api/observations")
    def clear_observations():
        store.clear()
        archive.clear_all()
        return "", 204

    return app


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

_TK_FILE_DIALOG_SCRIPT = """
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
path = filedialog.askopenfilename(
    title="RoadPush 분석 영상 선택",
    filetypes=[("동영상", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("모든 파일", "*.*")],
)
root.destroy()
print(path or "", end="")
"""


def _file_dialog_command() -> list[str]:
    """Return a native file-picker command for the server OS.

    Linux uses zenity when it is installed. Windows (and Linux without zenity)
    falls back to a tkinter dialog run in a separate Python process, because
    tkinter must not run inside Flask's worker threads.
    """
    if not sys.platform.startswith("win") and shutil.which("zenity"):
        return [
            "zenity",
            "--file-selection",
            "--title=RoadPush 분석 영상 선택",
            "--file-filter=동영상 | *.mp4 *.avi *.mov *.mkv *.m4v",
            "--file-filter=모든 파일 | *",
        ]
    return [sys.executable, "-c", _TK_FILE_DIALOG_SCRIPT]


def _file_sha256(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _inside_data_root(path_value: str | None, data_root: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).resolve()
    try:
        path.relative_to(data_root.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def _video_payload(video: dict[str, Any]) -> dict[str, Any]:
    result = dict(video)
    video_id = int(result["id"])
    result.update(
        observations_url=f"/api/videos/{video_id}/observations",
        segments_url=f"/api/videos/{video_id}/segments",
        source_url=f"/api/videos/{video_id}/source",
        report_url=f"/api/videos/{video_id}/report/download" if result.get("report_path") else None,
    )
    return result


def _observation_payload(item: dict[str, Any]) -> dict[str, Any]:
    result = dict(item)
    result["frame_url"] = f"/api/observations/{item['id']}/frame"
    result["segment_video_url"] = (
        f"/api/videos/{item['video_id']}/segments/{item['segment_index']}/video"
        if item.get("segment_video_path") else None
    )
    return result


def create_mysql_app(
    repository: MySQLRepository | None = None,
    data_root: str | Path | None = None,
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("ROADPUSH_MAX_UPLOAD_BYTES", str(2 * 1024**3)))
    root = Path(data_root or os.getenv("ROADPUSH_DATA_ROOT", Path(__file__).parent / "road_data"))
    root.mkdir(parents=True, exist_ok=True)
    uploads_root = root / "uploads"
    uploads_root.mkdir(parents=True, exist_ok=True)
    repo = repository or MySQLRepository()
    processor = VideoProcessor(repo, root)
    app.extensions["roadpush_repository"] = repo
    app.extensions["video_processor"] = processor
    app.config["ROADPUSH_DATA_ROOT"] = str(root)

    def start_video_processing(video: dict[str, Any]) -> None:
        thread = threading.Thread(
            target=processor.process,
            args=(int(video["id"]),),
            name=f"roadpush-video-{video['id']}",
            daemon=True,
        )
        thread.start()

    @app.errorhandler(RuntimeError)
    def handle_runtime_error(exc: RuntimeError):
        return jsonify({"error": str(exc)}), 503

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        try:
            repo.ping()
            return jsonify({"database": "connected"})
        except Exception as exc:
            return jsonify({"database": "unavailable", "error": str(exc)}), 503

    @app.get("/api/videos")
    def list_videos():
        return jsonify({"videos": [_video_payload(video) for video in repo.list_videos()]})

    @app.get("/api/map/observations")
    def get_latest_map_observations():
        try:
            limit = min(500_000, max(1, int(request.args.get("limit", 200_000))))
        except ValueError:
            return jsonify({"error": "limit은 정수여야 합니다."}), 400
        try:
            overlap_meters = max(
                0.0,
                float(os.getenv("ROADPUSH_GPS_OVERLAP_METERS", "5")),
            )
        except ValueError:
            overlap_meters = 5.0
        all_observations = repo.list_all_observations(limit=limit)
        observations = latest_overlapping_observations(all_observations, overlap_meters)
        return jsonify({
            "observations": [_observation_payload(item) for item in observations],
            "overlap_meters": overlap_meters,
            "superseded_count": len(all_observations) - len(observations),
        })

    @app.get("/api/videos/<int:video_id>")
    def get_video(video_id: int):
        video = repo.get_video(video_id)
        if not video:
            return jsonify({"error": "영상을 찾을 수 없습니다."}), 404
        return jsonify(_video_payload(video))

    @app.post("/api/videos/select-local")
    def select_local_video():
        if request.remote_addr not in {"127.0.0.1", "::1"}:
            return jsonify({"error": "로컬 파일 선택은 서버 PC의 localhost에서만 사용할 수 있습니다."}), 403
        repo.ensure_schema()
        try:
            dialog = subprocess.run(
                _file_dialog_command(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                timeout=600,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return jsonify({"error": f"로컬 파일 선택창을 열지 못했습니다: {exc}"}), 500
        if dialog.returncode != 0 or not dialog.stdout.strip():
            return jsonify({"cancelled": True}), 200

        source_video = Path(dialog.stdout.strip()).expanduser().resolve()
        if not source_video.is_file() or source_video.suffix.lower() not in VIDEO_EXTENSIONS:
            return jsonify({"error": "지원하는 영상 파일을 선택하세요."}), 400
        source_srt = next(
            (
                item for item in source_video.parent.iterdir()
                if item.is_file()
                and item.suffix.lower() == ".srt"
                and item.stem.casefold() == source_video.stem.casefold()
            ),
            None,
        )
        if source_srt is None:
            return jsonify({
                "error": f"같은 폴더에 {source_video.stem}.srt 파일이 없습니다."
            }), 400

        staging_directory = uploads_root / uuid.uuid4().hex
        staging_directory.mkdir(parents=True)
        safe_stem = secure_filename(source_video.stem) or "video"
        video_path = staging_directory / f"{safe_stem}{source_video.suffix.lower()}"
        srt_path = staging_directory / f"{safe_stem}.srt"
        try:
            shutil.copy2(source_video, video_path)
            shutil.copy2(source_srt, srt_path)
            from srt_gps import load_gps_srt

            load_gps_srt(srt_path)
            video, created = repo.create_video(
                original_name=source_video.name[:255],
                content_sha256=_file_sha256(video_path, srt_path),
                video_path=str(video_path),
                srt_path=str(srt_path),
            )
        except ValueError as exc:
            shutil.rmtree(staging_directory, ignore_errors=True)
            return jsonify({"error": str(exc)}), 400
        except Exception:
            shutil.rmtree(staging_directory, ignore_errors=True)
            raise

        if not created:
            shutil.rmtree(staging_directory, ignore_errors=True)
            return jsonify({
                "video": _video_payload(video),
                "duplicate": True,
                "srt_name": source_srt.name,
            }), 200
        start_video_processing(video)
        return jsonify({
            "video": _video_payload(video),
            "duplicate": False,
            "srt_name": source_srt.name,
        }), 202

    @app.post("/api/videos/upload")
    def upload_video():
        repo.ensure_schema()
        uploaded_video = request.files.get("video")
        uploaded_srt = request.files.get("srt")
        if not uploaded_video or not uploaded_srt:
            return jsonify({"error": "영상 파일과 같은 이름의 SRT 파일을 함께 선택하세요."}), 400

        original_video_name = (uploaded_video.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        original_srt_name = (uploaded_srt.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        video_suffix = Path(original_video_name).suffix.lower()
        srt_suffix = Path(original_srt_name).suffix.lower()
        if not original_video_name or video_suffix not in VIDEO_EXTENSIONS:
            return jsonify({"error": "지원하는 영상 형식은 MP4, AVI, MOV, MKV, M4V입니다."}), 400
        if srt_suffix != ".srt":
            return jsonify({"error": "GPS 자막은 SRT 파일이어야 합니다."}), 400
        if Path(original_video_name).stem.casefold() != Path(original_srt_name).stem.casefold():
            return jsonify({"error": "영상과 SRT 파일의 이름이 같아야 합니다."}), 400

        safe_stem = secure_filename(Path(original_video_name).stem) or "video"
        video_name = f"{safe_stem}{video_suffix}"
        srt_name = f"{safe_stem}.srt"

        staging_directory = uploads_root / uuid.uuid4().hex
        staging_directory.mkdir(parents=True)
        video_path = staging_directory / video_name
        srt_path = staging_directory / srt_name
        uploaded_video.save(video_path)
        uploaded_srt.save(srt_path)
        try:
            from srt_gps import load_gps_srt

            load_gps_srt(srt_path)
            content_hash = _file_sha256(video_path, srt_path)
            video, created = repo.create_video(
                original_name=original_video_name[:255],
                content_sha256=content_hash,
                video_path=str(video_path),
                srt_path=str(srt_path),
            )
        except ValueError as exc:
            shutil.rmtree(staging_directory, ignore_errors=True)
            return jsonify({"error": str(exc)}), 400
        except Exception:
            shutil.rmtree(staging_directory, ignore_errors=True)
            raise

        if not created:
            shutil.rmtree(staging_directory, ignore_errors=True)
            return jsonify({"video": _video_payload(video), "duplicate": True}), 200

        start_video_processing(video)
        return jsonify({"video": _video_payload(video), "duplicate": False}), 202

    @app.get("/api/videos/<int:video_id>/observations")
    def get_video_observations(video_id: int):
        if not repo.get_video(video_id):
            return jsonify({"error": "영상을 찾을 수 없습니다."}), 404
        try:
            limit = min(100_000, max(1, int(request.args.get("limit", 50_000))))
        except ValueError:
            return jsonify({"error": "limit은 정수여야 합니다."}), 400
        observations = repo.list_observations(video_id, limit=limit)
        return jsonify({"observations": [_observation_payload(item) for item in observations]})

    @app.get("/api/videos/<int:video_id>/segments")
    def get_video_segments(video_id: int):
        return jsonify({"segments": repo.list_segments(video_id)})

    @app.get("/api/observations/<int:observation_id>/frame")
    def get_observation_frame(observation_id: int):
        observation = repo.get_observation(observation_id)
        if not observation:
            return jsonify({"error": "프레임을 찾을 수 없습니다."}), 404
        path = _inside_data_root(observation.get("frame_path"), root)
        if not path:
            return jsonify({"error": "저장된 프레임 파일이 없습니다."}), 404
        return send_file(path, mimetype="image/jpeg", conditional=True)

    @app.get("/api/videos/<int:video_id>/source")
    def get_source_video(video_id: int):
        video = repo.get_video(video_id)
        path = _inside_data_root(video.get("video_path") if video else None, root)
        if not path:
            return jsonify({"error": "원본 영상이 없습니다."}), 404
        return send_file(path, conditional=True)

    @app.get("/api/videos/<int:video_id>/segments/<int:segment_index>/video")
    def get_segment_video(video_id: int, segment_index: int):
        segment = next(
            (item for item in repo.list_segments(video_id) if item["segment_index"] == segment_index),
            None,
        )
        path = _inside_data_root(segment.get("video_path") if segment else None, root)
        if not path:
            return jsonify({"error": "100m 구간 영상이 없습니다."}), 404
        return send_file(path, mimetype="video/mp4", conditional=True)

    @app.post("/api/videos/<int:video_id>/report")
    def make_report(video_id: int):
        try:
            path = create_inspection_report(repo, root, video_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({
            "message": "점검 리포트를 저장했습니다.",
            "filename": path.name,
            "download_url": f"/api/videos/{video_id}/report/download",
        })

    @app.get("/api/videos/<int:video_id>/report/download")
    def download_report(video_id: int):
        video = repo.get_video(video_id)
        path = _inside_data_root(video.get("report_path") if video else None, root)
        if not path:
            return jsonify({"error": "생성된 리포트가 없습니다."}), 404
        return send_file(path, mimetype="application/zip", as_attachment=True, download_name=path.name)

    return app


def create_app(
    segment_root: str | Path | None = None,
    encode_segment_videos: bool = False,
    repository: MySQLRepository | None = None,
    data_root: str | Path | None = None,
) -> Flask:
    # Existing unit tests and device-ingest compatibility keep their isolated archive.
    if segment_root is not None:
        return create_legacy_app(segment_root, encode_segment_videos)
    return create_mysql_app(repository=repository, data_root=data_root)


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
