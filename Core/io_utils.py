"""小型 JSON/文本产物的原子读写。"""

from __future__ import annotations

import errno
import json  # 将指标、摘要和运行清单序列化为标准 JSON。
import os  # os.replace 在同一文件系统内提供原子替换语义。
from pathlib import Path  # 统一处理 Windows/Unix 路径。
import tempfile  # 多进程共享目标时生成同目录唯一临时文件。
import time  # Windows/网络盘并发替换目标时做短暂重试。
from typing import Any  # JSON 入口允许字典、列表和标量等任意可序列化对象。


def lock_is_abandoned(path: str | Path, stale_seconds: float) -> bool:
    """判断 PID 锁是否已无主，避免 Ctrl+C 后等待数小时。"""

    target = Path(path)
    try:
        age = max(0.0, time.time() - target.stat().st_mtime)
    except FileNotFoundError:
        return False
    except OSError:
        return False
    try:
        raw_pid = target.read_text(encoding="ascii", errors="ignore").strip()
    except FileNotFoundError:
        return False
    except OSError:
        return age > float(stale_seconds)
    try:
        pid = int(raw_pid)
    except ValueError:
        # 创建锁和写入 PID 之间有极短窗口，不得抢走新锁。
        return age > 5.0
    if pid <= 0:
        return age > 5.0
    if os.name == "nt":
        # Windows 的 os.kill(pid, 0) 并不是 POSIX 式无副作用探测；
        # 用只读进程句柄判断，绝不向目标发送信号。
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        open_process.restype = ctypes.c_void_p
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
        handle = open_process(0x1000, False, pid)
        if handle:
            close_handle(handle)
            return False
        windows_error = ctypes.get_last_error()
        if windows_error in {87, 1168}:
            return True
        if windows_error == 5:
            return False
        return age > float(stale_seconds)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError as error:
        if error.errno in {errno.ESRCH, errno.EINVAL} or getattr(
            error, "winerror", None
        ) == 87:
            return True
        return age > float(stale_seconds)
    return False


def atomic_write_json(payload: Any, path: str | Path) -> Path:
    """以 UTF-8 和中文可读格式原子写入 JSON。"""

    # 将字符串和 Path 统一为 Path，后续所有路径操作保持平台无关。
    target = Path(path)
    # 阶段目录可能尚未建立，写文件前递归创建父目录。
    target.parent.mkdir(parents=True, exist_ok=True)
    # 每个进程必须拥有不同的临时文件。固定的 ``target.json.tmp`` 会在
    # E4 多 seed 并行时被另一个进程先 os.replace，造成 FileNotFoundError。
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.{os.getpid()}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        # ensure_ascii=False 保留中文，indent=2 便于人工检查实验结果。
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        # 只有完整写完临时文件后才覆盖目标，进程中断不会留下半个 JSON。
        # Linux 的 replace 是原子的；Windows/部分网络盘在另一进程刚完成
        # replace 的瞬间可能短暂返回 PermissionError，因此进行有界重试。
        for attempt in range(20):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_json(path: str | Path, default: Any = None) -> Any:
    """文件不存在时返回 default，存在但格式错误时保留异常。"""

    # 统一路径类型并只接受普通文件。
    target = Path(path)
    # 缺失是可预期状态，例如首次运行尚无 summary；直接返回调用方默认值。
    if not target.is_file():
        return default
    # JSON 损坏属于真实错误，不吞异常，以免继续使用不完整训练状态。
    return json.loads(target.read_text(encoding="utf-8"))
