from pathlib import Path


def mkdir(folder: str, exist_ok: bool = True, parents: bool = False) -> Path:
    folder: Path = Path(folder)
    folder.mkdir(exist_ok=exist_ok, parents=parents)
    return folder
