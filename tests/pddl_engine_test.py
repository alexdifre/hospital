import sys
import types

from core.task_planning.pddl_engine import (
    DEFAULT_PDDL_PLANNING_ENGINE,
    make_pddl_oneshot_planner,
    normalize_pddl_planning_engine,
)


def test_pddl_planning_engine_defaults_to_enhsp_opt():
    assert DEFAULT_PDDL_PLANNING_ENGINE == "enhsp-opt"
    assert normalize_pddl_planning_engine(None) == "enhsp-opt"


def test_pddl_planning_engine_uses_explicit_engine_name_without_aliases():
    assert normalize_pddl_planning_engine("enhsp-opt") == "enhsp-opt"
    assert normalize_pddl_planning_engine("enhsp") == "enhsp"


def test_make_pddl_oneshot_planner_requests_enhsp_opt(monkeypatch):
    captured = {}

    class FakeOneshotPlanner:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    fake_shortcuts = types.ModuleType("unified_planning.shortcuts")
    fake_shortcuts.OneshotPlanner = FakeOneshotPlanner
    monkeypatch.setitem(sys.modules, "unified_planning.shortcuts", fake_shortcuts)

    planner = make_pddl_oneshot_planner("enhsp-opt")

    assert isinstance(planner, FakeOneshotPlanner)
    assert captured["kwargs"]["name"] == "enhsp-opt"
