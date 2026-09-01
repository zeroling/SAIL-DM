"""控制台与文件双输出的阶段日志。"""

from __future__ import annotations

import logging  # Python 标准日志库，同时输出控制台和 UTF-8 文件。
import sys  # 显式使用 stdout，避免子进程恢复后继承到失效 stderr。
import time  # 网络盘短暂拒绝日志追加时进行一次短重试。
from pathlib import Path  # 平台无关地创建阶段日志目录。


class ResilientAppendHandler(logging.Handler):
    """不长期持有网络盘文件描述符的追加日志器。"""

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record) + "\n"
        except Exception:
            return
        # /gz-fs 上长期开启的 FileHandler 可在多层子进程恢复后
        # 变成 EBADF。每条日志重新打开并立即关闭，避免保留失效 fd。
        for attempt in range(2):
            try:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(message)
                return
            except OSError:
                if attempt == 0:
                    time.sleep(0.02)
        # 控制台内容已被外层 attempt 日志完整保留。网络盘
        # 连续两次追加失败时只放弃这一份重复 train.log，不打断训练。


def get_stage_logger(name: str, directory: str | Path) -> logging.Logger:
    """为一个阶段创建独立 UTF-8 日志，重复调用不会叠加 handler。"""

    # 每个训练阶段使用自己的目录，便于单独续训和排查。
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # 固定命名空间避免污染第三方库 root logger。
    logger = logging.getLogger(f"miccai.{name}")
    # 当前训练日志以 INFO 为主；异常仍由调用栈完整报告。
    logger.setLevel(logging.INFO)
    # 禁止向 root logger 传播，否则 PyCharm/脚本可能重复打印同一条消息。
    logger.propagate = False
    # 同一 Python 进程重复运行阶段时，先关闭并移除旧 handler，防止文件句柄泄漏。
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    # 时间、级别和正文三列足以重建训练过程，避免冗余模块名占据终端宽度。
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    # 控制台 handler 让用户实时观察训练。
    # 编排器本身使用 stdout 转发子进程日志。StreamHandler 默认
    # 选择 stderr，而 Linux 多层子进程在中断恢复后偶尔可能继承到
    # 已失效的 stderr 描述符，导致 logger.info 反复报 EBADF。stdout 是
    # 实时转发的唯一标准通道，因此显式绑定它。
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    # 文件日志仍用于恢复任务的进度排序，但不再长期占用网络盘 fd。
    file_handler = ResilientAppendHandler(directory / "train.log")
    file_handler.setFormatter(formatter)
    # 两个 handler 使用相同格式，屏幕与文件内容便于对应。
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
