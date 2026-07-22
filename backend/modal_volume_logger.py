"""
modal_volume_logger.py — Buffered request logger writing JSONL to a Modal
Volume (persistent storage across container restarts/scale-to-zero).

Cost: Modal Volumes are $0.09/GiB-month with 1 TiB/month free — a text log
like this will never leave the free tier.

Writes are buffered in memory and flushed to the mounted volume path
periodically, then the Volume is explicitly committed so the data is
durable and visible outside the current container. Without commit(),
writes only exist in this container's local view of the volume.
"""

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

logger = logging.getLogger("dheeren's_chat.volume_logger")


@dataclass
class RequestLogEntry:
    request_id:       str
    timestamp:        str
    endpoint:         str
    client_key:       str
    status:           str   # "ok" | "error" | "rate_limited_minute" | "rate_limited_day" | "busy" | "unauthorized"
    prompt:           Optional[str] = None
    system:           Optional[str] = None
    params:           Optional[dict] = None
    response:         Optional[str] = None
    tokens_generated: Optional[int] = None
    elapsed_ms:       Optional[float] = None
    error:            Optional[str] = None
    user_agent:       Optional[str] = None


class ModalVolumeRequestLogger:
    def __init__(self, log_dir: str, volume=None,
                 flush_interval_seconds: float = 30.0, max_buffer_size: int = 200):
        """
        log_dir: path INSIDE the container where the volume is mounted
                 (e.g. "/logs") — must match the mount path in modal_app.py.
        volume:  the modal.Volume object itself, passed in so we can call
                 .commit() after flushing. Optional — if None, writes still
                 happen locally but won't be explicitly committed (fine for
                 local dev without Modal).
        """
        self.log_dir = log_dir
        self.volume = volume
        self.flush_interval = flush_interval_seconds
        self.max_buffer_size = max_buffer_size
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, "requests.jsonl")

        self._buffer: List[RequestLogEntry] = []
        self._lock = threading.Lock()
        self._flush_task: Optional[asyncio.Task] = None

    def log(self, entry: RequestLogEntry) -> None:
        with self._lock:
            self._buffer.append(entry)
            should_flush_now = len(self._buffer) >= self.max_buffer_size
        if should_flush_now:
            threading.Thread(target=self._flush_sync, daemon=True).start()

    def _drain_buffer(self) -> List[RequestLogEntry]:
        with self._lock:
            drained, self._buffer = self._buffer, []
            return drained

    def _flush_sync(self) -> None:
        entries = self._drain_buffer()
        if not entries:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                for e in entries:
                    f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

            if self.volume is not None:
                self.volume.commit()   # make writes durable + visible elsewhere

            logger.info("[volume_logger] flushed %d entries to %s", len(entries), self.path)
        except Exception as exc:
            logger.error("[volume_logger] flush failed, re-buffering %d entries: %s",
                         len(entries), exc, exc_info=True)
            with self._lock:
                self._buffer = entries + self._buffer

    async def start_background_flush(self) -> None:
        async def loop():
            while True:
                await asyncio.sleep(self.flush_interval)
                await asyncio.to_thread(self._flush_sync)
        self._flush_task = asyncio.create_task(loop())

    async def stop_and_final_flush(self) -> None:
        if self._flush_task:
            self._flush_task.cancel()
        await asyncio.to_thread(self._flush_sync)

    @staticmethod
    def new_id() -> str:
        import uuid
        return uuid.uuid4().hex[:12]

    @staticmethod
    def now_iso() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"