"""Smoke test: every module in the skeleton imports."""

import importlib
import pkgutil

import sdc


def test_all_modules_import():
    for mod in pkgutil.walk_packages(sdc.__path__, prefix="sdc."):
        importlib.import_module(mod.name)
