#!/usr/bin/env python3
"""
Navigation Stack: Grid-based Global Planner + Local MPC
========================================================

Implements hierarchical navigation:
- OccupancyGrid: Discrete representation of environment
- AStarPlanner: Global path planning through rooms/around walls
- WaypointManager: Feeds short-range goals to MPC

This is the standard architecture used in ROS Navigation (move_base/Nav2)
and most commercial mobile robots.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import heapq


@dataclass
class GridCell:
    """A cell in the occupancy grid."""

    x: int  # Grid x index
    y: int  # Grid y index

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        # For heapq comparison
        return (self.x, self.y) < (other.x, other.y)


class OccupancyGrid:
    """
    2D occupancy grid representation of the environment.

    - 0 = free space
    - 1 = occupied (obstacle/wall)

    Handles conversion between world coordinates (meters) and grid indices.
    """

    def __init__(
        self,
        cell_size: float = 1.0,
        x_min: float = -5.0,
        x_max: float = 30.0,
        y_min: float = -15.0,
        y_max: float = 20.0,
    ):
        """
        Initialize occupancy grid.

        Args:
            cell_size: Size of each cell in meters
            x_min, x_max: World X bounds
            y_min, y_max: World Y bounds
        """
        self.cell_size = cell_size
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        # Calculate grid dimensions
        self.width = int(np.ceil((x_max - x_min) / cell_size))
        self.height = int(np.ceil((y_max - y_min) / cell_size))

        # Initialize grid (all free)
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)

        print(
            f"OccupancyGrid initialized: {self.width}x{self.height} cells "
            f"({cell_size}m resolution)"
        )
        print(f"  World bounds: X[{x_min}, {x_max}], Y[{y_min}, {y_max}]")

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (meters) to grid indices."""
        gx = int((x - self.x_min) / self.cell_size)
        gy = int((y - self.y_min) / self.cell_size)

        # Clamp to valid range
        gx = max(0, min(gx, self.width - 1))
        gy = max(0, min(gy, self.height - 1))

        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        x = self.x_min + (gx + 0.5) * self.cell_size
        y = self.y_min + (gy + 0.5) * self.cell_size
        return x, y

    def is_valid(self, gx: int, gy: int) -> bool:
        """Check if grid coordinates are valid."""
        return 0 <= gx < self.width and 0 <= gy < self.height

    def is_free(self, gx: int, gy: int) -> bool:
        """Check if cell is free (not occupied)."""
        if not self.is_valid(gx, gy):
            return False
        return self.grid[gy, gx] == 0

    def set_occupied(self, gx: int, gy: int):
        """Mark cell as occupied."""
        if self.is_valid(gx, gy):
            self.grid[gy, gx] = 1

    def set_free(self, gx: int, gy: int):
        """Mark cell as free."""
        if self.is_valid(gx, gy):
            self.grid[gy, gx] = 0

    def add_circular_obstacle(
        self,
        x: float,
        y: float,
        radius: float,
        inflation: float = 0.3,
    ):
        """
        Add a circular obstacle to the grid.

        Args:
            x, y: World coordinates of obstacle center
            radius: Obstacle radius in meters
            inflation: Extra margin for safety (robot radius)
        """
        total_radius = radius + inflation

        # Find grid cells that could be affected
        gx_center, gy_center = self.world_to_grid(x, y)
        cells_radius = int(np.ceil(total_radius / self.cell_size)) + 1

        for dy in range(-cells_radius, cells_radius + 1):
            for dx in range(-cells_radius, cells_radius + 1):
                gx = gx_center + dx
                gy = gy_center + dy

                if not self.is_valid(gx, gy):
                    continue

                # Check if cell center is within obstacle radius
                cell_x, cell_y = self.grid_to_world(gx, gy)
                dist = np.sqrt((cell_x - x) ** 2 + (cell_y - y) ** 2)

                if dist <= total_radius:
                    self.set_occupied(gx, gy)

    def add_rectangular_obstacle(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        inflation: float = 0.3,
    ):
        """
        Add a rectangular obstacle (wall) to the grid.

        Args:
            x_min, y_min: Bottom-left corner
            x_max, y_max: Top-right corner
            inflation: Extra margin for safety
        """
        # Inflate the rectangle
        x_min -= inflation
        y_min -= inflation
        x_max += inflation
        y_max += inflation

        # Convert to grid coordinates
        gx_min, gy_min = self.world_to_grid(x_min, y_min)
        gx_max, gy_max = self.world_to_grid(x_max, y_max)

        for gy in range(gy_min, gy_max + 1):
            for gx in range(gx_min, gx_max + 1):
                self.set_occupied(gx, gy)

    def add_wall(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        thickness: float = 0.1,
        inflation: float = 0.3,
    ):
        """
        Add a wall (line segment) to the grid.

        Args:
            x1, y1: Start point
            x2, y2: End point
            thickness: Wall thickness
            inflation: Extra margin
        """
        # Use Bresenham-like approach with thickness
        total_thickness = thickness + inflation

        # Sample points along the wall
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        num_samples = int(np.ceil(length / (self.cell_size * 0.5)))

        for i in range(num_samples + 1):
            t = i / max(num_samples, 1)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            self.add_circular_obstacle(x, y, total_thickness, inflation=0)

    def get_neighbors(self, gx: int, gy: int) -> List[Tuple[int, int, float]]:
        """
        Get valid neighboring cells for A*.

        Returns list of (gx, gy, cost) tuples.
        Uses 8-connectivity (diagonal moves allowed).
        """
        neighbors = []

        # 8 directions: N, NE, E, SE, S, SW, W, NW
        directions = [
            (0, 1, 1.0),  # N
            (1, 1, 1.414),  # NE
            (1, 0, 1.0),  # E
            (1, -1, 1.414),  # SE
            (0, -1, 1.0),  # S
            (-1, -1, 1.414),  # SW
            (-1, 0, 1.0),  # W
            (-1, 1, 1.414),  # NW
        ]

        for dx, dy, cost in directions:
            nx, ny = gx + dx, gy + dy
            if self.is_free(nx, ny):
                # Check diagonal clearance (prevent corner cutting)
                if abs(dx) == 1 and abs(dy) == 1:
                    # For diagonal moves, ensure both adjacent cells are free
                    if not self.is_free(gx + dx, gy) or not self.is_free(gx, gy + dy):
                        continue
                neighbors.append((nx, ny, cost))

        return neighbors

    def visualize(self, path: List[Tuple[int, int]] = None) -> str:
        """
        Create ASCII visualization of the grid.

        Args:
            path: Optional list of (gx, gy) cells to highlight
        """
        path_set = set(path) if path else set()

        lines = []
        for gy in range(self.height - 1, -1, -1):  # Top to bottom
            row = ""
            for gx in range(self.width):
                if (gx, gy) in path_set:
                    row += "* "
                elif self.grid[gy, gx] == 1:
                    row += "# "
                else:
                    row += ". "
            lines.append(row)

        return "\n".join(lines)


class AStarPlanner:
    """
    A* path planner on occupancy grid.

    Finds shortest path from start to goal while avoiding obstacles.
    """

    def __init__(self, grid: OccupancyGrid):
        """
        Initialize planner with a grid.

        Args:
            grid: OccupancyGrid instance
        """
        self.grid = grid

    def heuristic(self, gx: int, gy: int, goal_gx: int, goal_gy: int) -> float:
        """Euclidean distance heuristic."""
        return np.sqrt((gx - goal_gx) ** 2 + (gy - goal_gy) ** 2)

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        simplify: bool = True,
    ) -> Optional[List[np.ndarray]]:
        """
        Find path from start to goal.

        Args:
            start: Start position in world coordinates [x, y]
            goal: Goal position in world coordinates [x, y]
            simplify: If True, reduce waypoints using line-of-sight

        Returns:
            List of waypoints in world coordinates, or None if no path found
        """
        # Convert to grid coordinates
        start_gx, start_gy = self.grid.world_to_grid(start[0], start[1])
        goal_gx, goal_gy = self.grid.world_to_grid(goal[0], goal[1])

        # Check if start/goal are valid
        if not self.grid.is_free(start_gx, start_gy):
            print(f"  [A*] Start cell ({start_gx}, {start_gy}) is blocked!")
            # Try to find nearest free cell
            start_gx, start_gy = self._find_nearest_free(start_gx, start_gy)
            if start_gx is None:
                return None
            print(f"  [A*] Using nearest free cell: ({start_gx}, {start_gy})")

        if not self.grid.is_free(goal_gx, goal_gy):
            print(f"  [A*] Goal cell ({goal_gx}, {goal_gy}) is blocked!")
            goal_gx, goal_gy = self._find_nearest_free(goal_gx, goal_gy)
            if goal_gx is None:
                return None
            print(f"  [A*] Using nearest free cell: ({goal_gx}, {goal_gy})")

        # A* algorithm
        open_set = []  # Priority queue: (f_score, g_score, gx, gy)
        heapq.heappush(open_set, (0, 0, start_gx, start_gy))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {(start_gx, start_gy): 0}

        visited = set()

        while open_set:
            _, current_g, gx, gy = heapq.heappop(open_set)

            if (gx, gy) in visited:
                continue
            visited.add((gx, gy))

            # Goal reached?
            if gx == goal_gx and gy == goal_gy:
                # Reconstruct path
                path_grid = self._reconstruct_path(came_from, (gx, gy))

                # Convert to world coordinates
                path_world = [
                    np.array(self.grid.grid_to_world(px, py)) for px, py in path_grid
                ]

                # Simplify path (remove unnecessary waypoints)
                if simplify:
                    path_world = self._simplify_path(path_world)

                # Always include exact goal position
                path_world[-1] = goal.copy()

                return path_world

            # Explore neighbors
            for nx, ny, move_cost in self.grid.get_neighbors(gx, gy):
                if (nx, ny) in visited:
                    continue

                tentative_g = current_g + move_cost

                if tentative_g < g_score.get((nx, ny), float("inf")):
                    came_from[(nx, ny)] = (gx, gy)
                    g_score[(nx, ny)] = tentative_g
                    f_score = tentative_g + self.heuristic(nx, ny, goal_gx, goal_gy)
                    heapq.heappush(open_set, (f_score, tentative_g, nx, ny))

        # No path found
        print(
            f"  [A*] No path found from ({start_gx},{start_gy}) to ({goal_gx},{goal_gy})"
        )
        return None

    def _find_nearest_free(
        self, gx: int, gy: int, max_radius: int = 5
    ) -> Tuple[Optional[int], Optional[int]]:
        """Find nearest free cell using BFS."""
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:  # Only check boundary
                        nx, ny = gx + dx, gy + dy
                        if self.grid.is_free(nx, ny):
                            return nx, ny
        return None, None

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _simplify_path(
        self,
        path: List[np.ndarray],
        tolerance: float = 0.1,
        max_segment_length: float = 3.0,  # MPC-friendly segment limit
    ) -> List[np.ndarray]:
        """
        Simplify path by removing unnecessary waypoints.

        Uses line-of-sight checks on the grid.
        Also ensures no segment exceeds max_segment_length.
        """
        if len(path) <= 2:
            # Still need to check if the single segment is too long
            if len(path) == 2:
                return self._subdivide_long_segments(path, max_segment_length)
            return path

        simplified = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            # Try to skip ahead as far as possible
            farthest_visible = current_idx + 1

            for test_idx in range(current_idx + 2, len(path)):
                # Check line-of-sight
                if self._line_of_sight(path[current_idx], path[test_idx]):
                    # Also check segment length
                    segment_dist = np.linalg.norm(path[test_idx] - path[current_idx])
                    if segment_dist <= max_segment_length:
                        farthest_visible = test_idx
                    # If too long, stop extending

            simplified.append(path[farthest_visible])
            current_idx = farthest_visible

        # Final pass: subdivide any remaining long segments
        return self._subdivide_long_segments(simplified, max_segment_length)

    def _subdivide_long_segments(
        self,
        path: List[np.ndarray],
        max_length: float,
    ) -> List[np.ndarray]:
        """Break up any segment longer than max_length into smaller pieces."""
        result = [path[0]]

        for i in range(1, len(path)):
            prev = result[-1]
            curr = path[i]
            dist = np.linalg.norm(curr - prev)

            if dist > max_length:
                # Need to subdivide
                n_segments = int(np.ceil(dist / max_length))
                for j in range(1, n_segments):
                    t = j / n_segments
                    intermediate = prev + t * (curr - prev)
                    result.append(intermediate)

            result.append(curr)

        return result

    def _line_of_sight(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check if straight line between two points is clear."""
        # Sample points along the line
        dist = np.linalg.norm(p2 - p1)
        num_samples = int(np.ceil(dist / (self.grid.cell_size * 0.5)))

        for i in range(num_samples + 1):
            t = i / max(num_samples, 1)
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])

            gx, gy = self.grid.world_to_grid(x, y)
            if not self.grid.is_free(gx, gy):
                return False

        return True


class WaypointManager:
    """
    Manages waypoint following for the robot.

    Tracks current waypoint, provides next target for MPC,
    and determines when to advance to next waypoint.
    """

    def __init__(
        self,
        waypoints: List[np.ndarray],
        arrival_tolerance: float = 1.0,
        lookahead_distance: float = 2.0,
    ):
        """
        Initialize waypoint manager.

        Args:
            waypoints: List of waypoints from A* planner
            arrival_tolerance: Distance to waypoint to consider "arrived"
            lookahead_distance: How far ahead to set MPC target
        """
        self.waypoints = waypoints
        self.arrival_tolerance = arrival_tolerance
        self.lookahead_distance = lookahead_distance
        self.current_idx = 0

    @property
    def current_waypoint(self) -> Optional[np.ndarray]:
        """Get current target waypoint."""
        if self.current_idx < len(self.waypoints):
            return self.waypoints[self.current_idx]
        return None

    @property
    def is_complete(self) -> bool:
        """Check if all waypoints have been reached."""
        return self.current_idx >= len(self.waypoints)

    @property
    def final_goal(self) -> np.ndarray:
        """Get the final destination."""
        return self.waypoints[-1]

    def update(self, robot_pos: np.ndarray) -> np.ndarray:
        """
        Update waypoint tracking and get MPC target.

        Args:
            robot_pos: Current robot position [x, y]

        Returns:
            Target position for MPC
        """
        # Check if we've reached current waypoint
        while self.current_idx < len(self.waypoints):
            dist = np.linalg.norm(robot_pos - self.waypoints[self.current_idx])

            if dist < self.arrival_tolerance:
                self.current_idx += 1
                if self.current_idx < len(self.waypoints):
                    print(
                        f"    [WP] Reached waypoint {self.current_idx}/{len(self.waypoints)}"
                    )
            else:
                break

        if self.is_complete:
            return self.final_goal

        # Return current waypoint as target
        # (Could also implement pure pursuit / lookahead here)
        return self.waypoints[self.current_idx]

    def get_mpc_target(self, robot_pos: np.ndarray) -> np.ndarray:
        """
        Get target position for MPC.

        Uses lookahead along the path for smoother tracking.
        """
        if self.is_complete:
            return self.final_goal

        # Simple: just return current waypoint
        # More advanced: interpolate along path with lookahead
        return self.update(robot_pos)

    def reset(self):
        """Reset to start of waypoint list."""
        self.current_idx = 0

    @property
    def progress(self) -> float:
        """Get progress as fraction (0.0 to 1.0)."""
        if len(self.waypoints) == 0:
            return 1.0
        return self.current_idx / len(self.waypoints)


class NavigationStack:
    """
    Complete navigation stack integrating:
    - OccupancyGrid (environment representation)
    - A* Planner (global path planning)
    - WaypointManager (waypoint tracking)

    This feeds into MPC for local control.
    """

    def __init__(
        self,
        cell_size: float = 1.0,
        x_bounds: Tuple[float, float] = (-5.0, 30.0),
        y_bounds: Tuple[float, float] = (-15.0, 20.0),
        robot_radius: float = 0.3,
    ):
        """
        Initialize navigation stack.

        Args:
            cell_size: Grid cell size in meters
            x_bounds: (min, max) world X coordinates
            y_bounds: (min, max) world Y coordinates
            robot_radius: Robot radius for inflation
        """
        self.robot_radius = robot_radius

        # Create grid
        self.grid = OccupancyGrid(
            cell_size=cell_size,
            x_min=x_bounds[0],
            x_max=x_bounds[1],
            y_min=y_bounds[0],
            y_max=y_bounds[1],
        )

        # Create planner
        self.planner = AStarPlanner(self.grid)

        # Waypoint manager (created per navigation goal)
        self.waypoint_manager: Optional[WaypointManager] = None

        # Static obstacles (added once)
        self.static_obstacles: List[Dict] = []

    def add_obstacle(self, x: float, y: float, radius: float, name: str = ""):
        """Add a circular obstacle to the static map."""
        self.grid.add_circular_obstacle(x, y, radius, inflation=self.robot_radius)
        self.static_obstacles.append({"x": x, "y": y, "radius": radius, "name": name})

    def add_wall(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        thickness: float = 0.1,
    ):
        """Add a wall segment to the static map."""
        self.grid.add_wall(
            x1,
            y1,
            x2,
            y2,
            thickness=thickness,
            inflation=self.robot_radius,
        )

    def plan_to(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        arrival_tolerance: float = 1.0,
    ) -> bool:
        """
        Plan path from start to goal.

        Args:
            start: Start position [x, y]
            goal: Goal position [x, y]
            arrival_tolerance: How close to get to waypoints

        Returns:
            True if path found, False otherwise
        """
        print(
            f"  [Nav] Planning path: ({start[0]:.1f}, {start[1]:.1f}) → "
            f"({goal[0]:.1f}, {goal[1]:.1f})"
        )

        # Run A*
        waypoints = self.planner.plan(start, goal)

        if waypoints is None:
            print("  [Nav] No path found!")
            return False

        print(f"  [Nav] Path found: {len(waypoints)} waypoints")
        for i, wp in enumerate(waypoints):
            print(f"    WP{i}: ({wp[0]:.1f}, {wp[1]:.1f})")

        # Create waypoint manager
        self.waypoint_manager = WaypointManager(
            waypoints=waypoints,
            arrival_tolerance=arrival_tolerance,
        )

        return True

    def get_mpc_target(self, robot_pos: np.ndarray) -> Optional[np.ndarray]:
        """Get current target for MPC given robot position."""
        if self.waypoint_manager is None:
            return None
        return self.waypoint_manager.get_mpc_target(robot_pos)

    def is_navigation_complete(self) -> bool:
        """Check if robot has reached final goal."""
        if self.waypoint_manager is None:
            return True
        return self.waypoint_manager.is_complete

    def get_progress(self) -> float:
        """Get navigation progress (0.0 to 1.0)."""
        if self.waypoint_manager is None:
            return 1.0
        return self.waypoint_manager.progress

    def visualize_grid(self) -> str:
        """Get ASCII visualization of current grid."""
        if self.waypoint_manager is not None:
            # Show path on grid
            path_cells = [
                self.grid.world_to_grid(wp[0], wp[1])
                for wp in self.waypoint_manager.waypoints
            ]
            return self.grid.visualize(path_cells)
        return self.grid.visualize()


# =============================================================================
# Test Functions
# =============================================================================


def test_navigation_stack():
    """Test the navigation stack with a simple scenario."""
    print("=" * 60)
    print("Testing Navigation Stack")
    print("=" * 60)

    # Create navigation stack
    nav = NavigationStack(
        cell_size=1.0,
        x_bounds=(0.0, 20.0),
        y_bounds=(0.0, 20.0),
    )

    # Add some obstacles
    nav.add_obstacle(10, 10, radius=2.0, name="big_obstacle")
    nav.add_obstacle(5, 15, radius=1.0, name="small_obstacle")

    # Add a wall
    nav.add_wall(15, 5, 15, 15, thickness=0.2)

    # Plan path
    start = np.array([2.0, 2.0])
    goal = np.array([18.0, 18.0])

    success = nav.plan_to(start, goal)

    if success:
        print("\nGrid visualization (# = obstacle, * = path):")
        print(nav.visualize_grid())

        # Simulate robot movement
        print("\nSimulating navigation:")
        robot_pos = start.copy()

        for step in range(50):
            target = nav.get_mpc_target(robot_pos)
            if target is None or nav.is_navigation_complete():
                print(f"  Step {step}: Navigation complete!")
                break

            # Simple movement toward target
            direction = target - robot_pos
            dist = np.linalg.norm(direction)
            if dist > 0.5:
                robot_pos = robot_pos + 0.5 * direction / dist
            else:
                robot_pos = target

            if step % 5 == 0:
                print(
                    f"  Step {step}: Robot at ({robot_pos[0]:.1f}, {robot_pos[1]:.1f}), "
                    f"target ({target[0]:.1f}, {target[1]:.1f})"
                )

        print(f"\nFinal position: ({robot_pos[0]:.1f}, {robot_pos[1]:.1f})")
        print(f"Goal: ({goal[0]:.1f}, {goal[1]:.1f})")
        print(f"Distance to goal: {np.linalg.norm(robot_pos - goal):.2f}m")

    return success


if __name__ == "__main__":
    test_navigation_stack()
