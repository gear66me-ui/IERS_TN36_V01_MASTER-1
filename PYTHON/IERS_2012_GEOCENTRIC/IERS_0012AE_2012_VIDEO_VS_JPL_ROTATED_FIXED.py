# IERS-0012AE
# Audit reference: Python 3.12 dynamic-module registration fix for IERS-0012AD.
from __future__ import annotations

import importlib.machinery
import runpy
import sys
from pathlib import Path

VERSION = "IERS-0012AE"
AD_SCRIPT = Path("IERS_0012AD_2012_VIDEO_VS_JPL_ROTATED.py").resolve()


def install_python312_loader_fix() -> None:
    original_exec_module = importlib.machinery.SourceFileLoader.exec_module

    if getattr(original_exec_module, "_iers_0012ae_patched", False):
        return

    def exec_module_with_registration(loader, module):
        sys.modules.setdefault(module.__name__, module)
        return original_exec_module(loader, module)

    exec_module_with_registration._iers_0012ae_patched = True
    importlib.machinery.SourceFileLoader.exec_module = exec_module_with_registration


def main() -> int:
    if not AD_SCRIPT.exists():
        raise FileNotFoundError(f"Required IERS-0012AD script not found: {AD_SCRIPT}")

    install_python312_loader_fix()
    runpy.run_path(str(AD_SCRIPT), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# IERS-0012AE
