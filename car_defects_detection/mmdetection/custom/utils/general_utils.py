import os
from datetime import datetime


def make_file_logger(log_path):
    """
    Create a simple file logger(msg) that appends timestamped lines
    to the given log file path.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def logger(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")

    return logger