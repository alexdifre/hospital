"""
Obstacle geometry utilities for MPC planning.

Filters the full environment obstacle list down to the most relevant
obstacles for inclusion in the MPC problem (keeps it tractable).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List


def filter_nearby_obstacles(
    robot_pos: np.ndarray,
    goal_pos: np.ndarray,
    obstacles: List[Dict],
    max_distance: float = 10.0,
    max_obstacles: int = 3,
    safety_margin: float = 0.15,  # Add buffer to obstacle radius
) -> List[Dict]:
    """
    Filter to the most relevant obstacles for MPC.

    Priority: obstacles on or near the direct path to goal.
    This keeps the optimization problem tractable (3-5 obstacles max).

    Args:
        robot_pos: Current [x, y] position
        goal_pos: Target [x, y] position
        obstacles: Full list of obstacles [{x, y, radius}, ...]
        max_distance: Ignore obstacles further than this
        max_obstacles: Maximum number to return
        safety_margin: Buffer added to obstacle radius for planning

    Returns:
        Filtered list of most relevant obstacles (with inflated radii)
    """
    if len(obstacles) == 0:
        return []

    def point_to_segment_distance(
        point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray
    ) -> float:
        """Distance from point to line segment."""
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        if seg_len_sq < 1e-10:
            return np.linalg.norm(point - seg_start)
        t = np.clip(np.dot(point - seg_start, seg_vec) / seg_len_sq, 0, 1)
        projection = seg_start + t * seg_vec
        return np.linalg.norm(point - projection)

    scored_obstacles = []
    for obs in obstacles:
        obs_pos = np.array([obs["x"], obs["y"]])

        # Multiple distance metrics
        dist_to_robot = np.linalg.norm(obs_pos - robot_pos)
        dist_to_goal = np.linalg.norm(obs_pos - goal_pos)
        dist_to_path = point_to_segment_distance(obs_pos, robot_pos, goal_pos)

        # Effective distance: prioritize obstacles ON the path
        # An obstacle 2m from path is more dangerous than one 5m from robot but off-path
        effective_dist = min(dist_to_path, dist_to_robot * 0.5, dist_to_goal * 0.7)

        # Only include if within range (accounting for obstacle radius)
        if effective_dist < max_distance + obs["radius"]:
            scored_obstacles.append({**obs, "_score": effective_dist})

    # Sort by score (lower = more important)
    scored_obstacles.sort(key=lambda o: o["_score"])

    # Take top N, add safety margin, and remove scoring metadata
    filtered = []
    for obs in scored_obstacles[:max_obstacles]:
        filtered.append(
            {
                "x": obs["x"],
                "y": obs["y"],
                "radius": obs["radius"] + safety_margin,  # Inflate for safety
            }
        )

    return filtered
