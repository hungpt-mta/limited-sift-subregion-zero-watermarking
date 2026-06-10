from __future__ import annotations
import time
from datetime import datetime

def now_str() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def log(msg: str):
    print(f"[{now_str()}] {msg}", flush=True)

def fmt_eta(start_time: float, done: int, total: int) -> str:
    if done <= 0:
        return "elapsed=0.0 min, ETA=unknown"
    elapsed = time.perf_counter() - start_time
    rate = done / max(elapsed, 1e-9)
    remaining = max(0, total - done) / max(rate, 1e-9)
    return f"elapsed={elapsed/60:.1f} min, ETA={remaining/60:.1f} min, rate={rate:.2f} it/s"
