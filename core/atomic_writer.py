"""提供 JSON 文件的原子写入工具，通过临时文件加 os.replace 避免写入中断导致文件损坏。"""
import os
import json
import tempfile
from typing import Any


def atomic_write_json(filepath: str, data: Any, indent: int = 2) -> bool:
    """原子写入JSON文件（tmp + os.replace）"""
    try:
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dirname if dirname else '.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            os.replace(tmp_path, filepath)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        return True
    except Exception:
        return False
