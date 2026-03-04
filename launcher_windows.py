# Ouroboros — Windows Launcher
# Thin orchestrator: secrets, bootstrap, main loop.

import logging
import os, sys, json, time, uuid, pathlib, subprocess, datetime, threading, queue as _queue_mod
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

# ----------------------------
# 0) Install launcher deps
# ----------------------------
def install_launcher_deps() -> None:
    # Force UTF-8 encoding for Windows consoles
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        subprocess.run(["chcp", "65001"], check=False, capture_output=True)
    except Exception:
        pass
    print("1. Installation des dépendances...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "openai>=1.0.0", "requests", "python-dotenv", "jiter>=0.8.0"],
            check=True,
        )
    except Exception as e:
        print(f"⚠ Erreur lors de l'installation des dépendances: {e}")

def main():
    # ----------------------------
    # 1) Load environment
    # ----------------------------
    from dotenv import load_dotenv
    load_dotenv()

    DRIVE_ROOT = pathlib.Path("./data").resolve()
    REPO_DIR = pathlib.Path(".").resolve()

    # Create necessary directories
    for sub in ["state", "logs", "memory", "index", "locks", "archive"]:
        (DRIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 2) Config
    # ----------------------------
    TOTAL_BUDGET = float(os.environ.get("TOTAL_BUDGET", "50"))
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
    GITHUB_USER = os.environ.get("GITHUB_USER")
    GITHUB_REPO = os.environ.get("GITHUB_REPO")
    MAX_WORKERS = int(os.environ.get("OUROBOROS_MAX_WORKERS", "5"))

    # LLM Models
    MODEL_MAIN = os.environ.get("OUROBOROS_MODEL", "openrouter/free")
    MODEL_CODE = os.environ.get("OUROBOROS_MODEL_CODE", MODEL_MAIN)
    MODEL_LIGHT = os.environ.get("OUROBOROS_MODEL_LIGHT", MODEL_MAIN)

    os.environ["OUROBOROS_MODEL"] = MODEL_MAIN
    os.environ["OUROBOROS_MODEL_CODE"] = MODEL_CODE
    os.environ["OUROBOROS_MODEL_LIGHT"] = MODEL_LIGHT

    BRANCH_DEV = "ouroboros"
    BRANCH_STABLE = "ouroboros"

    # ----------------------------
    # 3) Initialize supervisor modules
    # ----------------------------
    from supervisor.state import (
        init as state_init, load_state, save_state, append_jsonl,
        update_budget_from_usage, status_text, rotate_chat_log_if_needed,
        init_state,
    )
    state_init(DRIVE_ROOT, TOTAL_BUDGET)
    init_state()

    from supervisor.telegram import (
        init as telegram_init, TelegramClient, send_with_budget,
    )
    if TELEGRAM_BOT_TOKEN:
        TG = TelegramClient(str(TELEGRAM_BOT_TOKEN))
        telegram_init(
            drive_root=DRIVE_ROOT,
            total_budget_limit=TOTAL_BUDGET,
            budget_report_every=10,
            tg_client=TG,
        )
    else:
        TG = None
        print("No TELEGRAM_BOT_TOKEN provided - running in local-only mode")

    from supervisor.git_ops import (
        init as git_ops_init, ensure_repo_present, safe_restart,
    )
    REMOTE_URL = None
    if GITHUB_TOKEN and GITHUB_USER and GITHUB_REPO:
        REMOTE_URL = f"https://{GITHUB_TOKEN}:x-oauth-basic@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"

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
    workers_init(
        repo_dir=REPO_DIR, drive_root=DRIVE_ROOT, max_workers=MAX_WORKERS,
        soft_timeout=600, hard_timeout=1800,
        total_budget_limit=TOTAL_BUDGET,
        branch_dev=BRANCH_DEV, branch_stable=BRANCH_STABLE,
    )

    from supervisor.events import dispatch_event

    # ----------------------------
    # 4) Bootstrap repo
    # ----------------------------
    print("2. Vérification du dépôt...")
    ensure_repo_present()

    # ----------------------------
    # 5) Start workers
    # ----------------------------
    print("3. Démarrage des workers...")
    kill_workers()
    spawn_workers(MAX_WORKERS)
    restored_pending = restore_pending_from_snapshot()
    persist_queue_snapshot(reason="startup")

    # ----------------------------
    # 6) Background consciousness
    # ----------------------------
    from ouroboros.consciousness import BackgroundConsciousness
    _consciousness = BackgroundConsciousness(
        drive_root=DRIVE_ROOT,
        repo_dir=REPO_DIR,
        event_queue=get_event_q(),
        owner_chat_id_fn=lambda: load_state().get("owner_chat_id"),
    )
    try:
        _consciousness.start()
        print("Background consciousness started.")
    except Exception as e:
        print(f"Consciousness start failed: {e}")

    print(f"Ouroboros lancé avec {MAX_WORKERS} workers. Tasks restaurées: {restored_pending}")

    # ----------------------------
    # 7) Main loop
    # ----------------------------
    import types
    _event_ctx = types.SimpleNamespace(
        DRIVE_ROOT=DRIVE_ROOT, REPO_DIR=REPO_DIR, BRANCH_DEV=BRANCH_DEV, BRANCH_STABLE=BRANCH_STABLE,
        TG=TG, WORKERS=WORKERS, PENDING=PENDING, RUNNING=RUNNING, MAX_WORKERS=MAX_WORKERS,
        send_with_budget=send_with_budget if TG else lambda *a, **k: None,
        load_state=load_state, save_state=save_state,
        update_budget_from_usage=update_budget_from_usage,
        append_jsonl=append_jsonl,
        enqueue_task=enqueue_task,
        cancel_task_by_id=cancel_task_by_id,
        queue_review_task=queue_review_task,
        persist_queue_snapshot=persist_queue_snapshot,
        safe_restart=safe_restart,
        kill_workers=kill_workers,
        spawn_workers=spawn_workers,
        sort_pending=sort_pending,
        consciousness=_consciousness,
    )

    offset = int(load_state().get("tg_offset") or 0)

    try:
        while True:
            ensure_workers_healthy()
            enforce_task_timeouts()
            enqueue_evolution_task_if_needed()
            assign_tasks()
            persist_queue_snapshot(reason="main_loop")

            # Drain worker events
            event_q = get_event_q()
            while not event_q.empty():
                try:
                    evt = event_q.get_nowait()
                    dispatch_event(evt, _event_ctx)
                except _queue_mod.Empty:
                    break

            # Telegram polling if TG is enabled
            if TG:
                try:
                    updates = TG.get_updates(offset=offset, timeout=5)
                    for upd in updates:
                        offset = int(upd["update_id"]) + 1
                        msg = upd.get("message") or upd.get("edited_message")
                        if not msg: continue
                        
                        chat_id = int(msg["chat"]["id"])
                        text = str(msg.get("text") or "")
                        
                        # Store owner if not set
                        st = load_state()
                        if st.get("owner_chat_id") is None:
                            st["owner_id"] = msg.get("from", {}).get("id")
                            st["owner_chat_id"] = chat_id
                            save_state(st)
                            send_with_budget(chat_id, "✅ Propriétaire enregistré.")
                        
                        if text.startswith("/status"):
                            send_with_budget(chat_id, status_text(WORKERS, PENDING, RUNNING, 600, 1800))
                        elif text:
                            # Forward to agent via threading to not block loop
                            threading.Thread(target=handle_chat_direct, args=(chat_id, text, None), daemon=True).start()
                    
                    # Update offset in state
                    st = load_state()
                    st["tg_offset"] = offset
                    save_state(st)
                except Exception as e:
                    log.debug(f"Telegram polling error: {e}")

            time.sleep(1)

    except KeyboardInterrupt:
        print("Arrêt d'Ouroboros...")
        kill_workers()
        _consciousness.stop()
        print("Ouroboros arrêté.")

if __name__ == "__main__":
    install_launcher_deps()
    main()
