"""
logger.py — Training logger

Writes to:
  logs/run_<timestamp>/train.log   — full text log
  logs/run_<timestamp>/metrics.csv — step,loss,lr,tok_per_sec,... (for plotting)
  logs/run_<timestamp>/config.json — full run config snapshot

Also prints to stdout with color.
"""

import os
import csv
import json
import logging
import datetime
from typing import Any, Dict, Optional


# ANSI colors for terminal
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_RED    = "\033[91m"
_DIM    = "\033[2m"


class TrainingLogger:
    """
    Unified logger for training runs.

    Usage:
        logger = TrainingLogger(run_name="gpt2-small-cosmopedia")
        logger.log_config(cfg_dict)
        logger.log_step(step=100, loss=3.2, lr=1e-4, tok_per_sec=850_000, ...)
        logger.log_eval(step=1000, val_loss=3.1)
        logger.log_checkpoint(step=1000, path="checkpoints/step_1000.pt")
        logger.close()
    """

    def __init__(self, run_name: Optional[str] = None, log_root: str = "logs"):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"run_{ts}"
        self.run_dir  = os.path.join(log_root, f"{self.run_name}_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)

        # ── Text log ────────────────────────────────────────
        log_path = os.path.join(self.run_dir, "train.log")
        self._file_logger = logging.getLogger(self.run_name)
        self._file_logger.setLevel(logging.DEBUG)
        self._file_logger.propagate = False

        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        self._file_logger.addHandler(fh)

        # ── CSV metrics ──────────────────────────────────────
        self._csv_path = os.path.join(self.run_dir, "metrics.csv")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = None          # initialized on first log_step (so header matches keys)
        self._csv_header_written = False
        self._fieldnames = ["step", "loss", "val_loss", "lr", "tok_per_sec", "grad_norm", "elapsed_sec", "tokens_seen", "step_ms"]

        self._info(f"Run directory : {self.run_dir}")
        self._info(f"Text log      : {log_path}")
        self._info(f"Metrics CSV   : {self._csv_path}")

    # ── Internal helpers ──────────────────────────────────────

    def _info(self, msg: str, color: str = _RESET):
        print(f"{color}{msg}{_RESET}")
        self._file_logger.info(msg)

    def _write_csv(self, row: Dict[str, Any]):
        if not self._csv_header_written:
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self._fieldnames,
                extrasaction='ignore'
            )
            self._csv_writer.writeheader()
            self._csv_header_written = True
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    # ── Public API ────────────────────────────────────────────

    def log_config(self, cfg: Dict[str, Any]):
        """Dump full config to JSON and to the text log."""
        cfg_path = os.path.join(self.run_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, default=str)

        self._info("\n" + "═" * 65, _BOLD)
        self._info(f"  Run : {self.run_name}", _BOLD)
        self._info("═" * 65, _BOLD)
        for k, v in cfg.items():
            self._info(f"  {k:<25} {v}")
        self._info("═" * 65 + "\n", _BOLD)

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        tok_per_sec: float,
        grad_norm: float,
        elapsed_sec: float,
        tokens_seen: int,
        step_ms: float,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Log a training step (called every N steps)."""
        row = {
            "step"       : step,
            "loss"       : round(loss, 6),
            "lr"         : f"{lr:.2e}",
            "tok_per_sec": round(tok_per_sec),
            "grad_norm"  : round(grad_norm, 4),
            "elapsed_sec": round(elapsed_sec, 1),
            "tokens_seen": tokens_seen,
            "step_ms"    : round(step_ms, 1),
        }
        if extra:
            row.update(extra)

        self._write_csv(row)

        # Pretty console line
        tokens_b = tokens_seen / 1e9
        msg = (
            f"step {step:>7,} | "
            f"loss {loss:>7.4f} | "
            f"lr {lr:.2e} | "
            f"{tok_per_sec/1e3:>7.1f}k tok/s | "
            f"gnorm {grad_norm:>5.2f} | "
            f"{tokens_b:.3f}B toks | "
            f"step_took {step_ms:>5.1f}ms | "
            f"total_time {elapsed_sec:>6.0f}s"
        )
        self._info(msg, _CYAN)

    def log_eval(self, step: int, val_loss: float, train_loss: Optional[float] = None):
        msg = f"{'─'*30}  EVAL step {step:,}  val_loss={val_loss:.4f}"
        if train_loss is not None:
            msg += f"  train_loss={train_loss:.4f}"
        msg += f"  {'─'*30}"
        self._info(msg, _GREEN)
        self._file_logger.info(f"EVAL step={step} val_loss={val_loss:.6f}")
    
        self._write_csv({
            "step"    : step,
            "val_loss": round(val_loss, 6),
        })

    def log_checkpoint(self, step: int, path: str):
        self._info(f"  ✓ Checkpoint saved → {path}  (step {step:,})", _YELLOW)
        self._file_logger.info(f"CHECKPOINT step={step} path={path}")

    def log_start(self, total_steps: int, total_tokens: int):
        self._info(
            f"\nTraining started  —  {total_steps:,} steps  |  "
            f"{total_tokens/1e9:.2f}B tokens\n",
            _BOLD,
        )

    def log_end(self, elapsed_sec: float, tokens_seen: int):
        h, r = divmod(int(elapsed_sec), 3600)
        m, s = divmod(r, 60)
        self._info(
            f"\n{'═'*65}\n"
            f"  Training complete  —  {tokens_seen/1e9:.3f}B tokens  "
            f"in {h}h {m}m {s}s\n"
            f"{'═'*65}\n",
            _GREEN,
        )

    def info(self, msg: str):
        """Generic info log."""
        self._info(msg)

    def warning(self, msg: str):
        self._info(f"WARNING: {msg}", _YELLOW)
        self._file_logger.warning(msg)

    def close(self):
        self._csv_file.close()
        for h in self._file_logger.handlers:
            h.close()
