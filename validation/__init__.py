"""环境校核与可信度报告（v4.3）。

    from validation import run_all_checks, build_trust_report, format_checks

    report = run_all_checks(seeds=(42, 7))
    print(format_checks(...))
    trust = build_trust_report(report)
    trust.write("output/validation")
"""

from validation.checks import (  # noqa: F401
    CheckResult,
    Checks,
    check_communication,
    check_fusion,
    check_geometry,
    check_measurement,
    format_checks,
    run_all_checks,
)
from validation.report import (  # noqa: F401
    DATA_CHAIN,
    LAYER_CN,
    UNSUPPORTED_CONCLUSIONS,
    UNVERIFIED_ASSUMPTIONS,
    TrustReport,
    build_trust_report,
)

__all__ = [
    "DATA_CHAIN", "LAYER_CN", "UNSUPPORTED_CONCLUSIONS", "UNVERIFIED_ASSUMPTIONS",
    "CheckResult", "Checks", "TrustReport", "build_trust_report",
    "check_communication", "check_fusion", "check_geometry", "check_measurement",
    "format_checks", "run_all_checks",
]
