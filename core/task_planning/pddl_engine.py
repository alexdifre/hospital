"""Unified Planning engine selection for PDDL task planning."""

from __future__ import annotations

from typing import Optional


DEFAULT_PDDL_PLANNING_ENGINE = "enhsp-opt"


def normalize_pddl_planning_engine(engine_name: Optional[str]) -> str:
    """Return the canonical Unified Planning engine name."""
    if engine_name is None:
        return DEFAULT_PDDL_PLANNING_ENGINE
    return engine_name.strip().lower()


def make_pddl_oneshot_planner(engine_name: Optional[str] = None):
    """Create a Unified Planning oneshot planner using the selected engine."""
    from unified_planning.shortcuts import OneshotPlanner

    return OneshotPlanner(name=normalize_pddl_planning_engine(engine_name))
