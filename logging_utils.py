"""统一日志工具。

目标：让所有脚本（仿真、训练、评测、AI 诊断）用同一套日志格式，
并同时输出到控制台与 `output/logs/<run>.log`，便于事后排查与论文留痕。

只用标准库 `logging`，不引入任何第三方依赖。
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import Optional

DEFAULT_LOG_DIR = os.path.join("output", "logs")

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_configured: bool = False
_file_handler: Optional[logging.FileHandler] = None


def ensure_utf8_console() -> None:
    """把 stdout/stderr 强制成 UTF-8（**幂等**，可重复调用）。

    为什么需要：Windows 控制台默认 GBK，而本工程的评测脚本会打印
    `⇒`、`⚠`、`✔`、`↔` 这类 GBK 不含的符号。不兜底的话，
    脚本往往**跑完全部计算、在最后打印判据时**才抛
    `UnicodeEncodeError`——整轮工作白费，而且看起来像"代码坏了"，
    实际只是控制台编码问题。已经踩过三次（`verify_v4.py`、
    `evaluate_cooperative_sensing.py`、`evaluate_observation_modes.py`）。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            # 已被重定向到不支持 reconfigure 的对象，或平台不支持
            pass


def setup_logging(
    run_name: str = "lpi_cogradar",
    log_dir: str = DEFAULT_LOG_DIR,
    level: int = logging.INFO,
    console: bool = True,
    log_to_file: bool = True,
) -> logging.Logger:
    """配置根日志器：控制台 + 文件。重复调用会替换文件处理器。"""
    global _configured, _file_handler

    ensure_utf8_console()

    root = logging.getLogger()
    root.setLevel(level)

    if not _configured:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        if console:
            stream = logging.StreamHandler(sys.stdout)
            stream.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
            root.addHandler(stream)
        _configured = True

    if log_to_file:
        if _file_handler is not None:
            root.removeHandler(_file_handler)
            _file_handler.close()
            _file_handler = None
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"{run_name}.log")
        _file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
        _file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        root.addHandler(_file_handler)
        root.info("日志文件：%s", os.path.abspath(path))

    return root


def get_logger(name: str) -> logging.Logger:
    """取一个带命名空间的日志器。"""
    return logging.getLogger(name)


def utc_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
