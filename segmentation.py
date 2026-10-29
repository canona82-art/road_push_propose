from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SEGMENT_DISTANCE_METERS = 100.0


def gps_distance_meters(start: Any, end: Any) -> float:
    lat1 = math.radians(start.latitude)
    lat2 = math.radians(end.latitude)
    delta_lat = lat2 - lat1
    delta_lng = math.radians(end.longitude - start.longitude)
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lng / 2) ** 2
    )
    return 12_742_000 * math.asin(math.sqrt(value))


def safe_video_directory(video_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", video_id).strip("._")
    return (cleaned or "video")[:80]


def frame_extension(mimetype: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
    }.get(mimetype.lower(), ".bin")


@dataclass
class ArchivedFrame:
    observation: Any
    filename: str


@dataclass
class ArchivedSegment:
    video_id: str
    index: int
    directory: Path
    frames: list[ArchivedFrame] = field(default_factory=list)
    distance_meters: float = 0.0
    finalized: bool = False
    video_file: str | None = None
    encoding_error: str | None = None


@dataclass
class VideoProgress:
    video_id: str
    directory: Path
    fps: float
    next_segment_index: int = 0
    current: ArchivedSegment | None = None
    closed: list[ArchivedSegment] = field(default_factory=list)


class SegmentArchive:
    """Incrementally saves frames in GPS-distance-based video segments."""

    def __init__(
        self,
        root: Path,
        distance_meters: float = SEGMENT_DISTANCE_METERS,
        encode_videos: bool = True,
        default_fps: float = 10.0,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.distance_meters = distance_meters
        self.encode_videos = encode_videos
        self.default_fps = default_fps
        self._videos: dict[str, VideoProgress] = {}

    def reset_video(self, video_id: str, fps: float | None = None) -> None:
        directory = self.root / safe_video_directory(video_id)
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        self._videos[video_id] = VideoProgress(
            video_id=video_id,
            directory=directory,
            fps=fps or self.default_fps,
        )

    def add_observation(
        self,
        observation: Any,
        frame_data: bytes,
        frame_mimetype: str,
        fps: float | None = None,
    ) -> int:
        video_id = observation.video_id
        if video_id not in self._videos:
            self.reset_video(video_id, fps=fps)
        progress = self._videos[video_id]
        if fps is not None:
            progress.fps = fps

        if progress.current is None:
            progress.current = self._new_segment(progress)
        segment = progress.current

        if segment.frames:
            segment.distance_meters += gps_distance_meters(
                segment.frames[-1].observation,
                observation,
            )

        local_index = len(segment.frames)
        extension = frame_extension(frame_mimetype)
        filename = f"frame_{local_index:06d}{extension}"
        frames_directory = segment.directory / "frames"
        frames_directory.mkdir(parents=True, exist_ok=True)
        (frames_directory / filename).write_bytes(frame_data)

        observation.segment_index = segment.index
        observation.segment_finalized = False
        segment.frames.append(ArchivedFrame(observation=observation, filename=filename))
        self._write_manifest(segment)

        if segment.distance_meters >= self.distance_meters:
            self._finalize_current(progress)

        self._refresh_manifests(progress)
        return observation.segment_index

    def finalize_video(self, video_id: str) -> list[dict[str, Any]]:
        progress = self._videos.get(video_id)
        if progress is None:
            return []
        if progress.current is not None and progress.current.frames:
            self._finalize_current(progress)
        self._refresh_manifests(progress)
        return self.list_segments(video_id)

    def list_segments(self, video_id: str) -> list[dict[str, Any]]:
        progress = self._videos.get(video_id)
        if progress is None:
            return []
        segments = [*progress.closed]
        if progress.current is not None and progress.current.frames:
            segments.append(progress.current)
        return [self._manifest(segment) for segment in segments]

    def video_path(self, video_id: str, segment_index: int) -> Path | None:
        progress = self._videos.get(video_id)
        if progress is None:
            return None
        for segment in progress.closed:
            if segment.index == segment_index and segment.video_file:
                path = segment.directory / segment.video_file
                return path if path.exists() else None
        return None

    def clear_all(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        self._videos.clear()

    def _new_segment(self, progress: VideoProgress) -> ArchivedSegment:
        index = progress.next_segment_index
        progress.next_segment_index += 1
        directory = progress.directory / f"segment_{index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        return ArchivedSegment(
            video_id=progress.video_id,
            index=index,
            directory=directory,
        )

    def _finalize_current(self, progress: VideoProgress) -> None:
        segment = progress.current
        if segment is None or not segment.frames:
            return
        segment.finalized = True
        for archived_frame in segment.frames:
            archived_frame.observation.segment_finalized = True
        if self.encode_videos:
            self._encode_video(segment, progress.fps)
        self._write_manifest(segment)
        progress.closed.append(segment)
        progress.current = None

    def _encode_video(self, segment: ArchivedSegment, fps: float) -> None:
        if any(frame.filename.endswith(".bin") for frame in segment.frames):
            segment.encoding_error = "Unsupported frame mimetype; frames were saved without MP4 encoding."
            return

        concat_path = segment.directory / "frames.ffconcat"
        duration = 1 / max(0.1, fps)
        lines = ["ffconcat version 1.0"]
        for archived_frame in segment.frames:
            frame_path = (segment.directory / "frames" / archived_frame.filename).resolve()
            escaped_path = str(frame_path).replace("'", "'\\''")
            lines.extend((f"file '{escaped_path}'", f"duration {duration:.9f}"))
        last_path = (segment.directory / "frames" / segment.frames[-1].filename).resolve()
        lines.append(f"file '{str(last_path).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'")
        concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        output_path = segment.directory / "video.mp4"
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-vf",
            f"fps={fps},scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            segment.encoding_error = str(exc)
            return
        if result.returncode == 0 and output_path.exists():
            segment.video_file = output_path.name
            segment.encoding_error = None
        else:
            segment.encoding_error = result.stderr[-500:] or "FFmpeg encoding failed."

    def _refresh_manifests(self, progress: VideoProgress) -> None:
        for segment in progress.closed:
            self._write_manifest(segment)
        if progress.current is not None and progress.current.frames:
            self._write_manifest(progress.current)

    def _manifest(self, segment: ArchivedSegment) -> dict[str, Any]:
        observations = []
        for archived_frame in segment.frames:
            item = archived_frame.observation.to_dict()
            item["stored_frame"] = f"frames/{archived_frame.filename}"
            observations.append(item)
        return {
            "video_id": segment.video_id,
            "segment_index": segment.index,
            "target_distance_meters": self.distance_meters,
            "distance_meters": round(segment.distance_meters, 3),
            "frame_count": len(segment.frames),
            "start_frame_index": observations[0]["frame_index"],
            "end_frame_index": observations[-1]["frame_index"],
            "video_level": max(
                (item["video_level"] for item in observations),
                key=lambda level: {"safe": 0, "caution": 1, "danger": 2}[level],
            ),
            "finalized": segment.finalized,
            "video_file": segment.video_file,
            "encoding_error": segment.encoding_error,
            "frames": observations,
        }

    def _write_manifest(self, segment: ArchivedSegment) -> None:
        manifest_path = segment.directory / "metadata.json"
        manifest_path.write_text(
            json.dumps(self._manifest(segment), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
