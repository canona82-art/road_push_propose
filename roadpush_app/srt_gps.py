from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GPSPoint:
    start: float
    end: float
    latitude: float
    longitude: float
    altitude: float = 0.0
    heading: float = 0.0
    captured_at: str = ""


def parse_srt_time(value: str) -> float:
    hours, minutes, seconds_millis = value.strip().split(":")
    seconds, millis = seconds_millis.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def load_gps_srt(path: str | Path) -> list[GPSPoint]:
    content = Path(path).read_text(encoding="utf-8", errors="ignore")
    entries: list[GPSPoint] = []
    for block in re.split(r"\r?\n\s*\r?\n", content.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        time_line = next((line for line in lines if "-->" in line), "")
        match = re.search(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            time_line,
        )
        if not match:
            continue
        gps_line = lines[-1]
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", gps_line)
        if len(numbers) < 2:
            continue
        time_index = lines.index(time_line)
        captured_at = lines[time_index + 1] if time_index + 2 < len(lines) else ""
        entries.append(
            GPSPoint(
                start=parse_srt_time(match.group(1)),
                end=parse_srt_time(match.group(2)),
                latitude=float(numbers[0]),
                longitude=float(numbers[1]),
                altitude=float(numbers[2]) if len(numbers) > 2 else 0.0,
                heading=float(numbers[3]) if len(numbers) > 3 else 0.0,
                captured_at=captured_at,
            )
        )
    if not entries:
        raise ValueError("SRT 파일에서 유효한 GPS 좌표를 찾지 못했습니다.")
    return entries


def gps_at_time(entries: list[GPSPoint], seconds: float) -> GPSPoint:
    if seconds <= entries[0].start:
        return entries[0]
    for index, point in enumerate(entries):
        if point.start <= seconds <= point.end:
            return point
        if index + 1 < len(entries) and point.end < seconds < entries[index + 1].start:
            return entries[index + 1]
    return entries[-1]

