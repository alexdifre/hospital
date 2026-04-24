#!/usr/bin/env python3
"""
Spatial A* Planner for Waypoint-Level Path Planning
===================================================

Grid-based A* search for collision-free path planning in 2D space.
Generates waypoint sequences that avoid obstacles.
"""

import numpy as np
import heapq
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field


@dataclass(order=True)
class GridNode:
    """Node in the A* search grid."""

    f_cost: float  # f = g + h
    g_cost: float = field(compare=False)
    h_cost: float = field(compare=False)
    x: int = field(compare=False)
    y: int = field(compare=False)
    parent: Optional["GridNode"] = field(default=None, compare=False)

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if not isinstance(other, GridNode):
            return False
        return self.x == other.x and self.y == other.y


class SpatialAStarPlanner:
    """
    Grid-based A* planner for 2D spatial path planning.

    Converts continuous world coordinates to discrete grid,
    performs A* search, and returns waypoint path.
    """

    def __init__(
        self,
        map_bounds_x: Tuple[float, float] = (-5.0, 30.0),
        map_bounds_y: Tuple[float, float] = (-20.0, 20.0),
        resolution: float = 0.5,
        robot_radius: float = 0.25,
        waypoint_spacing: float = 3.0,
    ):
        """
        Initialize spatial planner.

        Args:
            map_bounds_x: (min_x, max_x) in meters
            map_bounds_y: (min_y, max_y) in meters
            resolution: Grid cell size in meters
            robot_radius: Robot radius for safety margin
            waypoint_spacing: Distance between output waypoints in meters
        """

        self.map_bounds_x = map_bounds_x
        self.map_bounds_y = map_bounds_y
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.waypoint_spacing = waypoint_spacing

        # Grid dimensions
        self.width = int((map_bounds_x[1] - map_bounds_x[0]) / resolution)
        self.height = int((map_bounds_y[1] - map_bounds_y[0]) / resolution)

        # 8-connected grid movements (including diagonals)
        self.motions = [
            (1, 0, 1.0),  # Right
            (0, 1, 1.0),  # Up
            (-1, 0, 1.0),  # Left
            (0, -1, 1.0),  # Down
            (1, 1, 1.414),  # Diagonal up-right
            (-1, 1, 1.414),  # Diagonal up-left
            (1, -1, 1.414),  # Diagonal down-right
            (-1, -1, 1.414),  # Diagonal down-left
        ]

        print(f"SpatialAStarPlanner initialized")
        print(
            f"  Map bounds: x=[{map_bounds_x[0]:.1f}, {map_bounds_x[1]:.1f}], "
            f"y=[{map_bounds_y[0]:.1f}, {map_bounds_y[1]:.1f}]"
        )
        print(f"  Grid size: {self.width} x {self.height} cells")
        print(f"  Resolution: {resolution}m per cell")
        print(f"  Robot radius: {robot_radius}m + safety margin: {robot_radius}m")
        print(f"  Total safety clearance: {robot_radius * 2}m")
        print(f"  Waypoint spacing: {waypoint_spacing}m")

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int((x - self.map_bounds_x[0]) / self.resolution)
        gy = int((y - self.map_bounds_y[0]) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        x = self.map_bounds_x[0] + (gx + 0.5) * self.resolution
        y = self.map_bounds_y[0] + (gy + 0.5) * self.resolution
        return x, y

    def is_valid(
        self, gx: int, gy: int, obstacle_map: Optional[np.ndarray] = None
    ) -> bool:
        """Check if grid cell is valid (in bounds and not obstacle)."""

        # Check bounds
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return False

        # Check obstacles
        if obstacle_map is not None:
            if obstacle_map[gy, gx] > 0:
                return False

        return True

    def heuristic(self, gx1: int, gy1: int, gx2: int, gy2: int) -> float:
        """Euclidean distance heuristic."""
        dx = gx2 - gx1
        dy = gy2 - gy1
        return np.sqrt(dx * dx + dy * dy) * self.resolution

    def plan_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacle_map: Optional[np.ndarray] = None,
    ) -> Optional[Dict]:
        """
        Plan collision-free path from start to goal using A*.

        Args:
            start: Start position [x, y] in world coordinates
            goal: Goal position [x, y] in world coordinates
            obstacle_map: Binary obstacle map (height x width), 1 = obstacle

        Returns:
            Dict with:
                - waypoints: List of waypoint positions (densified)
                - path_length: Total path length in meters
                - straight_distance: Straight-line distance
                - nodes_expanded: Number of nodes explored
        """

        # Convert to grid coordinates
        start_gx, start_gy = self.world_to_grid(start[0], start[1])
        goal_gx, goal_gy = self.world_to_grid(goal[0], goal[1])

        # Check if start/goal are valid
        if not self.is_valid(start_gx, start_gy, obstacle_map):
            print(f"   [FAIL] Start position in collision")
            return None

        if not self.is_valid(goal_gx, goal_gy, obstacle_map):
            print(f"   [FAIL] Goal position in collision")
            return None

        # Calculate straight-line distance
        straight_distance = np.linalg.norm(goal - start)

        print(f"\n SPATIAL A* PLANNING")
        print(f"   Start: [{start[0]:.2f}, {start[1]:.2f}]")
        print(f"   Goal:  [{goal[0]:.2f}, {goal[1]:.2f}]")
        print(f"   Straight-line distance: {straight_distance:.2f}m")

        # A* search
        start_node = GridNode(
            f_cost=self.heuristic(start_gx, start_gy, goal_gx, goal_gy),
            g_cost=0.0,
            h_cost=self.heuristic(start_gx, start_gy, goal_gx, goal_gy),
            x=start_gx,
            y=start_gy,
            parent=None,
        )

        open_set = [start_node]
        closed_set = set()
        nodes_expanded = 0

        while open_set:
            # Get node with lowest f_cost
            current = heapq.heappop(open_set)

            # Check if already visited
            if (current.x, current.y) in closed_set:
                continue

            nodes_expanded += 1
            closed_set.add((current.x, current.y))

            # Check if goal reached
            if current.x == goal_gx and current.y == goal_gy:
                # Reconstruct path
                path = self._reconstruct_path(current)

                # Convert to world coordinates
                path_world = [self.grid_to_world(gx, gy) for gx, gy in path]

                # Calculate path length
                path_length = self._calculate_path_length(path_world)

                # Densify waypoints
                waypoints = self._densify_waypoints(path_world, self.waypoint_spacing)

                print(
                    f"   Path found: {len(waypoints)} waypoints, length {path_length:.2f}m"
                )
                print(f"    Explored {nodes_expanded} nodes")

                return {
                    "waypoints": waypoints,
                    "path_length": path_length,
                    "straight_distance": straight_distance,
                    "nodes_expanded": nodes_expanded,
                    "success": True,
                }

            # Expand neighbors
            for dx, dy, cost in self.motions:
                next_x = current.x + dx
                next_y = current.y + dy

                # Check validity
                if not self.is_valid(next_x, next_y, obstacle_map):
                    continue

                # Check if already visited
                if (next_x, next_y) in closed_set:
                    continue

                # Calculate costs
                g_cost = current.g_cost + cost * self.resolution
                h_cost = self.heuristic(next_x, next_y, goal_gx, goal_gy)
                f_cost = g_cost + h_cost

                # Create successor node
                successor = GridNode(
                    f_cost=f_cost,
                    g_cost=g_cost,
                    h_cost=h_cost,
                    x=next_x,
                    y=next_y,
                    parent=current,
                )

                heapq.heappush(open_set, successor)

        # No path found
        print(f"   [FAIL] No path found (expanded {nodes_expanded} nodes)")
        return None

    def _reconstruct_path(self, goal_node: GridNode) -> List[Tuple[int, int]]:
        """Reconstruct path from goal node to start."""
        path = []
        current = goal_node

        while current is not None:
            path.append((current.x, current.y))
            current = current.parent

        path.reverse()
        return path

    def _calculate_path_length(self, path_world: List[Tuple[float, float]]) -> float:
        """Calculate total path length."""
        length = 0.0
        for i in range(len(path_world) - 1):
            dx = path_world[i + 1][0] - path_world[i][0]
            dy = path_world[i + 1][1] - path_world[i][1]
            length += np.sqrt(dx * dx + dy * dy)
        return length

    def _densify_waypoints(
        self, path_world: List[Tuple[float, float]], spacing: float
    ) -> List[np.ndarray]:
        """
        Densify path with waypoints at regular intervals.

        Args:
            path_world: Original path from A*
            spacing: Desired spacing between waypoints

        Returns:
            List of waypoint positions as numpy arrays
        """

        if len(path_world) == 0:
            return []

        waypoints = [np.array(path_world[0])]

        for i in range(len(path_world) - 1):
            start = np.array(path_world[i])
            end = np.array(path_world[i + 1])

            # Calculate segment length and direction
            segment = end - start
            segment_length = np.linalg.norm(segment)

            if segment_length < 1e-6:
                continue

            direction = segment / segment_length

            # Add waypoints along segment
            num_waypoints = int(np.ceil(segment_length / spacing))
            for j in range(1, num_waypoints):
                t = j / num_waypoints
                waypoint = start + t * segment
                waypoints.append(waypoint)

        # Always add final goal
        waypoints.append(np.array(path_world[-1]))

        return waypoints


def test_spatial_planner():
    """Test the spatial A* planner."""

    print("Testing Spatial A* Planner")
    print("=" * 80)

    # Create planner
    planner = SpatialAStarPlanner(
        map_bounds_x=(-5.0, 30.0),
        map_bounds_y=(-20.0, 20.0),
        resolution=0.5,
        robot_radius=0.25,
        waypoint_spacing=3.0,
    )

    # Test 1: Simple path without obstacles
    print("\n" + "=" * 80)
    print("TEST 1: Simple path (no obstacles)")
    print("=" * 80)

    start = np.array([0.0, 0.0])
    goal = np.array([10.0, 10.0])

    result = planner.plan_path(start, goal, obstacle_map=None)

    if result:
        print(f" Path found!")
        print(f"   Waypoints: {len(result['waypoints'])}")
        print(f"   Path length: {result['path_length']:.2f}m")
        print(f"   Straight distance: {result['straight_distance']:.2f}m")
        print(f"   Detour: {result['path_length'] - result['straight_distance']:.2f}m")

    # Test 2: Path with obstacle
    print("\n" + "=" * 80)
    print("TEST 2: Path with obstacle")
    print("=" * 80)

    # Create obstacle map
    obstacle_map = np.zeros((planner.height, planner.width))

    # Add a wall obstacle in the middle
    wall_gx, wall_gy = planner.world_to_grid(5.0, 0.0)
    for dy in range(-10, 10):
        gy = wall_gy + dy
        if 0 <= gy < planner.height:
            obstacle_map[gy, wall_gx] = 1

    start = np.array([0.0, 0.0])
    goal = np.array([10.0, 0.0])

    result = planner.plan_path(start, goal, obstacle_map=obstacle_map)

    if result:
        print(f" Path found (routing around obstacle)!")
        print(f"   Waypoints: {len(result['waypoints'])}")
        print(f"   Path length: {result['path_length']:.2f}m")
        print(f"   Straight distance: {result['straight_distance']:.2f}m")
        print(f"   Detour: {result['path_length'] - result['straight_distance']:.2f}m")

    print("\n Spatial A* Planner test complete!")


if __name__ == "__main__":
    test_spatial_planner()
