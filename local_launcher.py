# Local launcher for Ouroboros (adapted from colab_launcher.py for local development)
# Removes Colab-specific dependencies and uses local paths

import logging
import os, sys, json, time, uuid, pathlib, subprocess, datetime, threading, queue as _queue_mod
from typing import Any, Dict, List, Optional, Set, Tuple

# Required for Windows multiprocessing with spawn
from multiprocessing import freeze_support

log = logging.getLogger(__name__)

# Install launcher deps
def install_launcher_deps() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "openai>=1.0.0", "requests"],
        check=True,
    )

install_launcher_deps()

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Set local DRIVE_ROOT
DRIVE_ROOT = pathlib.Path("./data").resolve()

# Create necessary directories
for sub in ["state", "logs", "memory", "index", "locks", "archive"]:
    (DRIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)

# Set default env vars if not set
import platform
_DEFAULT_WORKER_METHOD = "spawn" if platform.system() == "Windows" else "fork"
os.environ.setdefault("OUROBOROS_WORKER_START_METHOD", _DEFAULT_WORKER_METHOD)
os.environ.setdefault("OUROBOROS_DIAG_HEARTBEAT_SEC", "30")
os.environ.setdefault("OUROBOROS_DIAG_SLOW_CYCLE_SEC", "20")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Assume env vars are set in .env or environment
# Required: OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN, TOTAL_BUDGET, GITHUB_TOKEN, GITHUB_USER, GITHUB_REPO

from ouroboros.llm import DEFAULT_LIGHT_MODEL

# Initialize supervisor modules
from supervisor.state import (
    init as state_init, load_state, save_state, append_jsonl,
    update_budget_from_usage, status_text, rotate_chat_log_if_needed,
    init_state,
)
state_init(DRIVE_ROOT, float(os.environ.get("TOTAL_BUDGET", "50")))
init_state()

# Skip Telegram for local (no TG client)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if TELEGRAM_BOT_TOKEN:
    from supervisor.telegram import (
        init as telegram_init, TelegramClient, send_with_budget,
    )
    TG = TelegramClient(str(TELEGRAM_BOT_TOKEN))
    telegram_init(
        drive_root=DRIVE_ROOT,
        total_budget_limit=float(os.environ.get("TOTAL_BUDGET", "50")),
        budget_report_every=10,
        tg_client=TG,
    )
else:
    print("No TELEGRAM_BOT_TOKEN provided - running in local-only mode (no Telegram interface)")

from supervisor.git_ops import (
    init as git_ops_init, ensure_repo_present, checkout_and_reset,
    sync_runtime_dependencies, import_test, safe_restart,
)
REPO_DIR = pathlib.Path(".").resolve()  # Current dir as repo
REMOTE_URL = f"https://{os.environ['GITHUB_TOKEN']}:x-oauth-basic@github.com/{os.environ['GITHUB_USER']}/{os.environ['GITHUB_REPO']}.git"
BRANCH_DEV = "ouroboros"
BRANCH_STABLE = "ouroboros-stable"
git_ops_init(
    repo_dir=REPO_DIR, drive_root=DRIVE_ROOT, remote_url=REMOTE_URL,
    branch_dev=BRANCH_DEV, branch_stable=BRANCH_STABLE,
)

from supervisor.queue import (
    enqueue_task, enforce_task_timeouts, enqueue_evolution_task_if_needed,
    persist_queue_snapshot, restore_pending_from_snapshot,
    cancel_task_by_id, queue_review_task, sort_pending,
)

from supervisor.workers import (
    init as workers_init, get_event_q, WORKERS, PENDING, RUNNING,
    spawn_workers, kill_workers, assign_tasks, ensure_workers_healthy,
    handle_chat_direct, _get_chat_agent, auto_resume_after_restart,
)
MAX_WORKERS = int(os.environ.get("OUROBOROS_MAX_WORKERS", "5"))
SOFT_TIMEOUT_SEC = 600
HARD_TIMEOUT_SEC = 1800
workers_init(
    repo_dir=REPO_DIR, drive_root=DRIVE_ROOT, max_workers=MAX_WORKERS,
    soft_timeout=SOFT_TIMEOUT_SEC, hard_timeout=HARD_TIMEOUT_SEC,
    total_budget_limit=float(os.environ.get("TOTAL_BUDGET", "50")),
    branch_dev=BRANCH_DEV, branch_stable=BRANCH_STABLE,
)

from supervisor.events import dispatch_event

# Bootstrap repo
ensure_repo_present()
ok, msg = safe_restart(reason="bootstrap", unsynced_policy="rescue_and_reset")
if not ok:
    print(f"Bootstrap failed: {msg}")
    sys.exit(1)

# Start workers
if __name__ == '__main__':
    freeze_support()
    
    kill_workers()
    spawn_workers(MAX_WORKERS)
    restored_pending = restore_pending_from_snapshot()
    persist_queue_snapshot(reason="startup")

    print(f"Ouroboros launched locally with {MAX_WORKERS} workers. Restored {restored_pending} pending tasks.")

    # Start background consciousness
    from ouroboros.consciousness import BackgroundConsciousness

    _consciousness = BackgroundConsciousness(
        drive_root=DRIVE_ROOT,
        repo_dir=REPO_DIR,
        event_queue=get_event_q(),
        owner_chat_id_fn=lambda: None,  # No Telegram
    )

    try:
        _consciousness.start()
        print("Background consciousness started.")
    except Exception as e:
        print(f"Consciousness start failed: {e}")

    # Simple local loop (no Telegram polling)
    print("Ouroboros is running locally. Use /status, /bg, etc. via console input (not implemented yet). Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
            ensure_workers_healthy()
            enforce_task_timeouts()
            enqueue_evolution_task_if_needed()
            assign_tasks()
            persist_queue_snapshot(reason="main_loop")
    except KeyboardInterrupt:
        print("Stopping Ouroboros...")
        kill_workers()
        _consciousness.stop()
        print("Ouroboros stopped.")
