import time
from datetime import datetime
from datetime import timezone


def get_current_time(format: str ="%y-%m-%d,%H:%M:%S") -> str:
    return datetime.now().strftime(f"UTC{time.timezone / -(60*60):+},{format}")

def get_current_utc_time(format: str ="%y-%m-%d,%H:%M:%S") -> str:
    return datetime.now(timezone.utc).strftime(f"{format}")
