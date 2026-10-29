from __future__ import annotations

import math
import os
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from segmentation import gps_distance_meters
from srt_gps import GPSPoint, gps_at_time, load_gps_srt


ANALYSIS_LOCK = threading.Lock()


def risk_level(score: float, grade: str = "") -> str:
    if "DISCONNECTED" in grade:
        return "danger"
    if score <= 25:
        return "safe"
    if score <= 50:
        return "caution"
    return "danger"


@dataclass
class SegmentState:
    id: int
    index: int
    directory: Path
    start_distance: float
    start_seconds: float
    end_seconds: float
    risks: list[int] = field(default_factory=list)
    frame_count: int = 0


class _Location:
    def __init__(self, point: GPSPoint) -> None:
        self.latitude = point.latitude
        self.longitude = point.longitude


class VideoProcessor:
    def __init__(self, repository: Any, data_root: str | Path) -> None:
        self.repository = repository
        self.data_root = Path(data_root)

    def process(self, video_id: int) -> None:
        with ANALYSIS_LOCK:
            self._process_locked(video_id)

    def _process_locked(self, video_id: int) -> None:
        video = self.repository.get_video(video_id)
        if not video:
            return
        self.repository.update_video(video_id, status="processing", progress=0, error_message=None)
        try:
            import cv2
            from detecting_ai.roadpush_core_f import RoadPushAnalyzer

            project_root = Path(__file__).resolve().parent
            analyzer = RoadPushAnalyzer(
                seg_weight=str(project_root / "detecting_ai/roadpush_weights/sidewalk_seg_kr/weights/best.pt"),
                det_weight=str(project_root / "detecting_ai/roadpush_weights/obstacle_det_kr/weights/best.pt"),
            )
            analyzer.reset_metrics()
            gps_entries = load_gps_srt(video["srt_path"])
            capture = cv2.VideoCapture(video["video_path"])
            if not capture.isOpened():
                raise RuntimeError("영상 파일을 열 수 없습니다.")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0 or math.isnan(fps):
                fps = 30.0
            total_frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            duration = total_frames / fps
            requested_analysis_fps = float(os.getenv("ROADPUSH_ANALYSIS_FPS", "5"))
            requested_analysis_fps = max(0.5, min(fps, requested_analysis_fps))
            frame_step = max(1, round(fps / requested_analysis_fps))
            analysis_fps = fps / frame_step
            self.repository.update_video(
                video_id,
                fps=fps,
                total_frames=total_frames,
                duration_seconds=duration,
            )

            video_directory = self.data_root / "videos" / str(video_id)
            segments_directory = video_directory / "segments"
            segments_directory.mkdir(parents=True, exist_ok=True)

            previous_location: _Location | None = None
            cumulative_distance = 0.0
            all_risks: list[int] = []
            segments: dict[int, SegmentState] = {}
            pending_observations: list[dict[str, Any]] = []
            source_frame_index = 0
            analyzed_frame_count = 0

            while capture.isOpened():
                ok, frame = capture.read()
                if not ok:
                    break
                if source_frame_index % frame_step:
                    source_frame_index += 1
                    if source_frame_index % max(10, frame_step * 5) == 0:
                        self.repository.update_video(
                            video_id,
                            processed_frames=source_frame_index,
                            progress=min(99, int(source_frame_index * 100 / total_frames)),
                            distance_meters=cumulative_distance,
                        )
                    continue

                seconds = source_frame_index / fps
                gps = gps_at_time(gps_entries, seconds)
                location = _Location(gps)
                if previous_location is not None:
                    step = gps_distance_meters(previous_location, location)
                    # Ignore impossible one-frame GPS jumps while retaining normal walking/driving data.
                    if step <= 50:
                        cumulative_distance += step
                previous_location = location
                segment_index = int(cumulative_distance // 100.0)

                state = segments.get(segment_index)
                if state is None:
                    segment_directory = segments_directory / f"segment_{segment_index:03d}"
                    (segment_directory / "frames").mkdir(parents=True, exist_ok=True)
                    segment_id = self.repository.create_segment(
                        video_id, segment_index, cumulative_distance, seconds
                    )
                    state = SegmentState(
                        id=segment_id,
                        index=segment_index,
                        directory=segment_directory,
                        start_distance=segment_index * 100.0,
                        start_seconds=seconds,
                        end_seconds=seconds,
                    )
                    segments[segment_index] = state

                analyzed_frame, effective_ratio, risk_score, grade = analyzer.process_frame(
                    frame, enable_blur=True
                )
                risk_score = int(risk_score)
                level = risk_level(risk_score, grade)
                local_index = state.frame_count
                frame_path = state.directory / "frames" / f"frame_{local_index:06d}.jpg"
                if not cv2.imwrite(str(frame_path), analyzed_frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                    raise RuntimeError(f"분석 프레임을 저장하지 못했습니다: {frame_path.name}")

                pending_observations.append(dict(
                    video_id=video_id,
                    segment_id=state.id,
                    frame_index=source_frame_index,
                    video_seconds=seconds,
                    latitude=gps.latitude,
                    longitude=gps.longitude,
                    altitude=gps.altitude,
                    heading=gps.heading,
                    effective_ratio=float(effective_ratio),
                    risk_score=risk_score,
                    level=level,
                    grade=grade,
                    captured_at=gps.captured_at,
                    frame_path=str(frame_path),
                ))
                analyzed_frame_count += 1
                if analyzed_frame_count == 1 or len(pending_observations) >= 10:
                    self.repository.add_observations(pending_observations)
                    pending_observations.clear()
                state.risks.append(risk_score)
                state.frame_count += 1
                state.end_seconds = seconds
                all_risks.append(risk_score)
                source_frame_index += 1

                if analyzed_frame_count == 1 or analyzed_frame_count % 5 == 0:
                    self.repository.update_video(
                        video_id,
                        processed_frames=source_frame_index,
                        progress=min(99, int(source_frame_index * 100 / total_frames)),
                        distance_meters=cumulative_distance,
                    )
            capture.release()
            self.repository.add_observations(pending_observations)

            for state in segments.values():
                segment_video = state.directory / "video.mp4"
                self._encode_segment(state.directory / "frames", segment_video, analysis_fps)
                average_risk = sum(state.risks) / max(1, len(state.risks))
                max_risk = max(state.risks, default=0)
                self.repository.finish_segment(
                    state.id,
                    end_distance_meters=min(cumulative_distance, (state.index + 1) * 100.0),
                    distance_meters=max(
                        0.0,
                        min(cumulative_distance, (state.index + 1) * 100.0) - state.start_distance,
                    ),
                    end_seconds=state.end_seconds,
                    average_risk=average_risk,
                    max_risk=max_risk,
                    safety_score=max(0.0, 100.0 - average_risk),
                    level=risk_level(max_risk),
                    frame_count=state.frame_count,
                    video_path=str(segment_video) if segment_video.exists() else None,
                )

            average_risk = sum(all_risks) / max(1, len(all_risks))
            max_risk = max(all_risks, default=0)
            self.repository.update_video(
                video_id,
                status="completed",
                progress=100,
                processed_frames=source_frame_index,
                distance_meters=cumulative_distance,
                safety_score=max(0.0, 100.0 - average_risk),
                overall_level=risk_level(max_risk),
                completed_at=datetime.now(),
            )
        except Exception as exc:
            self.repository.update_video(
                video_id,
                status="failed",
                error_message=str(exc)[:2000],
            )

    @staticmethod
    def _encode_segment(frames_directory: Path, output_path: Path, fps: float) -> None:
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
            "-start_number", "0", "-i", str(frames_directory / "frame_%06d.jpg"),
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
        if result.returncode != 0:
            output_path.unlink(missing_ok=True)

