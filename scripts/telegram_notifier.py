#!/usr/bin/env python3
"""Send Telegram updates for active SLURM jobs and their logs.

Usage examples:
    TM_TELEGRAM_BOT_TOKEN=xxx TM_TELEGRAM_CHAT_ID=123 \
        python scripts/telegram_notifier.py

    TM_SLURM_LOG_DIR=/path/to/slurm_logs \
        python scripts/telegram_notifier.py --prefix custom-prefix

Environment variables (can also be passed as flags):
    TM_TELEGRAM_BOT_TOKEN   Telegram bot token.
    TM_TELEGRAM_CHAT_ID     Telegram chat id.
    TM_SLURM_LOG_DIR        Directory containing Slurm log files.
    TM_SLURM_USER           User to query in squeue (defaults to $USER).
    TM_NOTIFY_POLL_SECS     Poll interval in seconds (default 60).
    TM_NOTIFY_PREFIXES      Comma-separated log prefixes to watch.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, Set, Tuple

import requests

DEFAULT_LOG_PREFIXES: Tuple[str, ...] = (
    "tm-eiie-batch",
    "tm-eiie-optuna",
    "tm-eiie-sweep",
    "tm-eiie-sweep-retry",
    "tm-eiie-dual-sweep",
)

TELEGRAM_MAX = 4000  # Telegram hard limit is 4096 characters; keep a margin.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send Telegram notifications for active slurm jobs and log output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bot-token",
        default=os.getenv("TM_TELEGRAM_BOT_TOKEN"),
        help="Telegram bot token (or set TM_TELEGRAM_BOT_TOKEN)",
    )
    parser.add_argument(
        "--chat-id",
        default=os.getenv("TM_TELEGRAM_CHAT_ID"),
        help="Telegram chat id (or set TM_TELEGRAM_CHAT_ID)",
    )
    parser.add_argument(
        "--log-dir",
        default=os.getenv(
            "TM_SLURM_LOG_DIR",
            str(Path(__file__).resolve().parent.parent / "slurm_logs"),
        ),
        help="Directory containing slurm log files",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("TM_SLURM_USER") or os.getenv("USER") or "laperia",
        help="User to query in squeue",
    )
    parser.add_argument(
        "--poll-secs",
        type=int,
        default=int(os.getenv("TM_NOTIFY_POLL_SECS", "60")),
        help="Polling interval in seconds",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        help="Log filename prefix to monitor; can repeat (fallback: defaults or TM_NOTIFY_PREFIXES)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print messages instead of sending to Telegram",
    )
    return parser.parse_args()


def build_prefixes(raw_prefixes: Iterable[str] | None) -> Tuple[str, ...]:
    if raw_prefixes:
        return tuple(p.strip() for p in raw_prefixes if p and p.strip())
    env_value = os.getenv("TM_NOTIFY_PREFIXES")
    if env_value:
        return tuple(p.strip() for p in env_value.split(",") if p.strip())
    return DEFAULT_LOG_PREFIXES


def send(msg: str, *, bot_token: str, chat_id: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] {msg}")
        return
    if not bot_token or not chat_id:
        print("[telegram_notifier] Missing bot token or chat id; nothing sent.")
        return

    trimmed = msg[-TELEGRAM_MAX:]
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data={"chat_id": chat_id, "text": trimmed},
            timeout=10,
        )
        if not resp.ok:
            print(
                f"[telegram_notifier] Telegram send failed: HTTP {resp.status_code}",
                flush=True,
            )
    except requests.RequestException as exc:
        print(f"[telegram_notifier] Telegram send failed: {type(exc).__name__}", flush=True)


def tail_file(path: Path, last_pos: int = 0):
    with path.open("r") as f:
        f.seek(last_pos)
        data = f.read()
        return data, f.tell()


def get_active_jobs(user: str):
    try:
        res = subprocess.run(
            ["squeue", "-u", user, "-o", "%i|%A|%T", "-h"],
            capture_output=True,
            text=True,
            check=True,
        )
        text = res.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return {}, f"squeue error: {exc}", set()
    jobs: Dict[str, str] = {}
    active_job_ids: Set[str] = set()
    if text:
        for line in text.splitlines():
            parts = line.split("|")
            if len(parts) < 3:
                continue
            display_id = parts[0].strip()
            real_id = parts[1].strip()
            state = parts[2].strip()
            if real_id.isdigit():
                active_job_ids.add(real_id)
            jobs[display_id] = state
    return jobs, text, active_job_ids


def _job_id_from_log_name(path: Path):
    m = re.search(r"-(\d+)\.(?:out|err)$", path.name)
    return m.group(1) if m else None


def iter_candidate_logs(
    active_job_ids: Set[str], prefixes: Tuple[str, ...], log_dir: Path
) -> Iterator[Path]:
    for prefix in prefixes:
        for path in log_dir.glob(f"{prefix}-*.out"):
            job_id = _job_id_from_log_name(path)
            if job_id and job_id in active_job_ids:
                yield path
        for path in log_dir.glob(f"{prefix}-*.err"):
            job_id = _job_id_from_log_name(path)
            if job_id and job_id in active_job_ids:
                yield path


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    prefixes = build_prefixes(args.prefix)

    last_positions: Dict[Path, int] = {}
    last_squeue_snapshot: str | None = None

    if not log_dir.exists():
        print(f"[telegram_notifier] Log dir not found: {log_dir}")
        return

    while True:
        jobs, squeue_snapshot, active_job_ids = get_active_jobs(args.user)
        if squeue_snapshot != last_squeue_snapshot:
            send(
                "📋 SQUEUE\n" + (squeue_snapshot if squeue_snapshot else "No jobs in queue."),
                bot_token=args.bot_token,
                chat_id=args.chat_id or "",
                dry_run=args.dry_run,
            )
            last_squeue_snapshot = squeue_snapshot

        current_paths = set(iter_candidate_logs(active_job_ids, prefixes, log_dir))
        for path in current_paths:
            if not path.exists():
                continue
            if path not in last_positions:
                last_positions[path] = path.stat().st_size
                continue
            data, last_positions[path] = tail_file(path, last_positions[path])
            if data:
                icon = "⚠️" if path.suffix == ".err" else "💻"
                send(
                    f"{icon} {path.name}\n{data[-TELEGRAM_MAX:]}",
                    bot_token=args.bot_token,
                    chat_id=args.chat_id or "",
                    dry_run=args.dry_run,
                )

        # Stop tracking files once their jobs are no longer active.
        for path in list(last_positions):
            if path not in current_paths:
                last_positions.pop(path, None)

        time.sleep(args.poll_secs)


if __name__ == "__main__":
    main()
