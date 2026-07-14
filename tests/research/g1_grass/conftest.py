from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_DIR = (
    Path(__file__).resolve().parents[3]
    / "source"
    / "unitree_rl_lab"
    / "unitree_rl_lab"
    / "tasks"
    / "locomotion"
    / "research"
    / "g1_grass"
)


def load_module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODULE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def pytest_configure():
    load_module("risk_curriculum")
    load_module("curriculum_runtime")
    load_module("risk_metrics")
