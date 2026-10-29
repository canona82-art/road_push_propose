from __future__ import annotations

import os
import re
import threading
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterator


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS videos (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
        original_name VARCHAR(255) NOT NULL,
        content_sha256 CHAR(64) NOT NULL UNIQUE,
        video_path VARCHAR(1024) NOT NULL,
        srt_path VARCHAR(1024) NOT NULL,
        status ENUM('queued', 'processing', 'completed', 'failed') NOT NULL DEFAULT 'queued',
        progress TINYINT UNSIGNED NOT NULL DEFAULT 0,
        error_message TEXT NULL,
        fps DOUBLE NULL,
        total_frames INT UNSIGNED NOT NULL DEFAULT 0,
        processed_frames INT UNSIGNED NOT NULL DEFAULT 0,
        duration_seconds DOUBLE NOT NULL DEFAULT 0,
        distance_meters DOUBLE NOT NULL DEFAULT 0,
        safety_score DOUBLE NULL,
        overall_level ENUM('safe', 'caution', 'danger') NULL,
        report_path VARCHAR(1024) NULL,
        created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        completed_at DATETIME(6) NULL,
        INDEX idx_videos_created_at (created_at),
        INDEX idx_videos_status (status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS segments (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
        video_id BIGINT UNSIGNED NOT NULL,
        segment_index INT UNSIGNED NOT NULL,
        start_distance_meters DOUBLE NOT NULL DEFAULT 0,
        end_distance_meters DOUBLE NOT NULL DEFAULT 0,
        distance_meters DOUBLE NOT NULL DEFAULT 0,
        start_seconds DOUBLE NOT NULL DEFAULT 0,
        end_seconds DOUBLE NOT NULL DEFAULT 0,
        average_risk DOUBLE NOT NULL DEFAULT 0,
        max_risk TINYINT UNSIGNED NOT NULL DEFAULT 0,
        safety_score DOUBLE NOT NULL DEFAULT 100,
        level ENUM('safe', 'caution', 'danger') NOT NULL DEFAULT 'safe',
        frame_count INT UNSIGNED NOT NULL DEFAULT 0,
        video_path VARCHAR(1024) NULL,
        created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        UNIQUE KEY uq_video_segment (video_id, segment_index),
        CONSTRAINT fk_segments_video FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
        INDEX idx_segments_video (video_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS observations (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
        video_id BIGINT UNSIGNED NOT NULL,
        segment_id BIGINT UNSIGNED NOT NULL,
        frame_index INT UNSIGNED NOT NULL,
        video_seconds DOUBLE NOT NULL,
        latitude DOUBLE NOT NULL,
        longitude DOUBLE NOT NULL,
        altitude DOUBLE NOT NULL DEFAULT 0,
        heading DOUBLE NOT NULL DEFAULT 0,
        effective_ratio DOUBLE NOT NULL DEFAULT 0,
        risk_score TINYINT UNSIGNED NOT NULL,
        level ENUM('safe', 'caution', 'danger') NOT NULL,
        grade VARCHAR(80) NOT NULL,
        captured_at VARCHAR(128) NULL,
        frame_path VARCHAR(1024) NOT NULL,
        created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        UNIQUE KEY uq_video_frame (video_id, frame_index),
        CONSTRAINT fk_observations_video FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
        CONSTRAINT fk_observations_segment FOREIGN KEY (segment_id) REFERENCES segments(id) ON DELETE CASCADE,
        INDEX idx_observations_video_frame (video_id, frame_index),
        INDEX idx_observations_segment (segment_id),
        INDEX idx_observations_level (level)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def serialize_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: _json_value(value) for key, value in row.items()}


class MySQLRepository:
    """Small MySQL data access layer; connections are created per worker thread."""

    def __init__(self) -> None:
        self.host = os.getenv("ROADPUSH_DB_HOST", "127.0.0.1")
        self.port = int(os.getenv("ROADPUSH_DB_PORT", "3306"))
        self.user = os.getenv("ROADPUSH_DB_USER", "root")
        self.password = os.getenv("ROADPUSH_DB_PASSWORD", "1234")
        self.database = os.getenv("ROADPUSH_DB_NAME", "road_push")
        if not re.fullmatch(r"[A-Za-z0-9_]+", self.database):
            raise ValueError("ROADPUSH_DB_NAME에는 영문, 숫자, 밑줄만 사용할 수 있습니다.")
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    @staticmethod
    def _driver():
        try:
            import pymysql
        except ImportError as exc:
            raise RuntimeError(
                "PyMySQL이 설치되지 않았습니다. "
                "python -m pip install PyMySQL 명령으로 설치하세요."
            ) from exc
        return pymysql

    def _connect(self, with_database: bool = True):
        pymysql = self._driver()
        kwargs = {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "charset": "utf8mb4",
            "autocommit": False,
            "cursorclass": pymysql.cursors.DictCursor,
            "connect_timeout": 5,
            "read_timeout": 30,
            "write_timeout": 30,
        }
        if with_database:
            kwargs["database"] = self.database
        try:
            return pymysql.connect(**kwargs)
        except pymysql.MySQLError as exc:
            raise RuntimeError(
                f"MySQL에 연결할 수 없습니다 ({self.host}:{self.port}, 사용자 {self.user}). "
                "MySQL 서버 실행 상태와 ROADPUSH_DB_* 환경변수를 확인하세요."
            ) from exc

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            connection = self._connect(with_database=False)
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"CREATE DATABASE IF NOT EXISTS `{self.database}` "
                        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                    )
                connection.commit()
            finally:
                connection.close()

            connection = self._connect()
            try:
                with connection.cursor() as cursor:
                    for statement in SCHEMA_STATEMENTS:
                        cursor.execute(statement)
                connection.commit()
                self._schema_ready = True
            finally:
                connection.close()

    @contextmanager
    def connection(self) -> Iterator[Any]:
        self.ensure_schema()
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def ping(self) -> None:
        with self.connection() as connection:
            connection.ping(reconnect=True)

    def create_video(
        self,
        original_name: str,
        content_sha256: str,
        video_path: str,
        srt_path: str,
    ) -> tuple[dict[str, Any], bool]:
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM videos WHERE content_sha256=%s",
                    (content_sha256,),
                )
                existing = cursor.fetchone()
                if existing:
                    return serialize_row(existing) or {}, False
                cursor.execute(
                    """
                    INSERT INTO videos (original_name, content_sha256, video_path, srt_path)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (original_name, content_sha256, video_path, srt_path),
                )
                video_id = cursor.lastrowid
            connection.commit()
        return self.get_video(video_id) or {}, True

    def get_video(self, video_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM videos WHERE id=%s", (video_id,))
                return serialize_row(cursor.fetchone())

    def list_videos(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM videos ORDER BY created_at DESC, id DESC")
                return [serialize_row(row) or {} for row in cursor.fetchall()]

    def update_video(self, video_id: int, **values: Any) -> None:
        allowed = {
            "status", "progress", "error_message", "fps", "total_frames",
            "processed_frames", "duration_seconds", "distance_meters",
            "safety_score", "overall_level", "report_path", "completed_at",
        }
        selected = {key: value for key, value in values.items() if key in allowed}
        if not selected:
            return
        assignments = ", ".join(f"{key}=%s" for key in selected)
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE videos SET {assignments} WHERE id=%s",
                    (*selected.values(), video_id),
                )
            connection.commit()

    def create_segment(self, video_id: int, segment_index: int, start_distance: float, start_seconds: float) -> int:
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO segments
                        (video_id, segment_index, start_distance_meters, end_distance_meters,
                         start_seconds, end_seconds)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE id=LAST_INSERT_ID(id)
                    """,
                    (video_id, segment_index, start_distance, start_distance, start_seconds, start_seconds),
                )
                segment_id = cursor.lastrowid
            connection.commit()
        return int(segment_id)

    def finish_segment(self, segment_id: int, **values: Any) -> None:
        allowed = {
            "end_distance_meters", "distance_meters", "end_seconds", "average_risk",
            "max_risk", "safety_score", "level", "frame_count", "video_path",
        }
        selected = {key: value for key, value in values.items() if key in allowed}
        assignments = ", ".join(f"{key}=%s" for key in selected)
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE segments SET {assignments} WHERE id=%s",
                    (*selected.values(), segment_id),
                )
            connection.commit()

    def add_observation(self, **values: Any) -> int:
        columns = (
            "video_id", "segment_id", "frame_index", "video_seconds", "latitude",
            "longitude", "altitude", "heading", "effective_ratio", "risk_score",
            "level", "grade", "captured_at", "frame_path",
        )
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"INSERT INTO observations ({', '.join(columns)}) VALUES "
                    f"({', '.join(['%s'] * len(columns))})",
                    tuple(values[column] for column in columns),
                )
                observation_id = cursor.lastrowid
            connection.commit()
        return int(observation_id)

    def add_observations(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        columns = (
            "video_id", "segment_id", "frame_index", "video_seconds", "latitude",
            "longitude", "altitude", "heading", "effective_ratio", "risk_score",
            "level", "grade", "captured_at", "frame_path",
        )
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"INSERT INTO observations ({', '.join(columns)}) VALUES "
                    f"({', '.join(['%s'] * len(columns))})",
                    [tuple(row[column] for column in columns) for row in rows],
                )
            connection.commit()

    def list_observations(self, video_id: int, limit: int = 50_000) -> list[dict[str, Any]]:
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT o.*, s.segment_index, s.video_path AS segment_video_path
                    FROM observations o
                    JOIN segments s ON s.id=o.segment_id
                    WHERE o.video_id=%s
                    ORDER BY o.frame_index
                    LIMIT %s
                    """,
                    (video_id, limit),
                )
                return [serialize_row(row) or {} for row in cursor.fetchall()]

    def list_all_observations(self, limit: int = 200_000) -> list[dict[str, Any]]:
        """Return map observations newest-video first, including source metadata."""
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT o.*, s.segment_index, s.level AS segment_level,
                           s.video_path AS segment_video_path,
                           v.original_name AS video_name,
                           v.created_at AS video_created_at
                    FROM observations o
                    JOIN segments s ON s.id=o.segment_id
                    JOIN videos v ON v.id=o.video_id
                    WHERE v.status IN ('processing', 'completed')
                    ORDER BY v.created_at DESC, v.id DESC, o.frame_index
                    LIMIT %s
                    """,
                    (limit,),
                )
                return [serialize_row(row) or {} for row in cursor.fetchall()]

    def list_previous_observations(
        self,
        video_id: int,
        limit: int = 200_000,
    ) -> list[dict[str, Any]]:
        """Return observations from inspections uploaded before ``video_id``."""
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT o.*, s.segment_index, s.video_path AS segment_video_path,
                           v.original_name AS video_name,
                           v.created_at AS video_created_at
                    FROM observations o
                    JOIN segments s ON s.id=o.segment_id
                    JOIN videos v ON v.id=o.video_id
                    JOIN videos current_video ON current_video.id=%s
                    WHERE v.status='completed'
                      AND (
                        v.created_at < current_video.created_at
                        OR (v.created_at = current_video.created_at AND v.id < current_video.id)
                      )
                    ORDER BY v.created_at DESC, v.id DESC, o.frame_index
                    LIMIT %s
                    """,
                    (video_id, limit),
                )
                return [serialize_row(row) or {} for row in cursor.fetchall()]

    def get_observation(self, observation_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT o.*, s.segment_index, s.video_path AS segment_video_path
                    FROM observations o
                    JOIN segments s ON s.id=o.segment_id
                    WHERE o.id=%s
                    """,
                    (observation_id,),
                )
                return serialize_row(cursor.fetchone())

    def list_segments(self, video_id: int) -> list[dict[str, Any]]:
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM segments WHERE video_id=%s ORDER BY segment_index",
                    (video_id,),
                )
                return [serialize_row(row) or {} for row in cursor.fetchall()]

