import os
from pathlib import Path


def load_global_schema():
    if not os.path.exists(Path("runs") / "global_schema.txt"):
        print("Global Schema not exist")
        exit()
    with open(Path("runs") / "global_schema.txt", encoding="utf-8") as file:
        return file.read()
