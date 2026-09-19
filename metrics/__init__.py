"""指标与报告包。

collector 负责逐步/汇总指标与 CSV；
report   负责 HTML 实验结果报告。
"""

from .collector import (
    STEP_FIELDS,
    SUMMARY_FIELDS,
    extract_step_metrics,
    format_summary_table,
    summarize_run,
    write_step_metrics_csv,
    write_summary_csv,
)
from .report import build_curves_html, build_html_report, write_curves_html, write_html_report

__all__ = [
    "STEP_FIELDS",
    "SUMMARY_FIELDS",
    "extract_step_metrics",
    "summarize_run",
    "format_summary_table",
    "write_step_metrics_csv",
    "write_summary_csv",
    "build_html_report",
    "write_html_report",
    "build_curves_html",
    "write_curves_html",
]
