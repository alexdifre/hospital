#!/usr/bin/env python3
"""
Expanded Hospital Ward MuJoCo Environment
==========================================

50x50m hospital ward with 14 PDDL-aligned locations for medication delivery and meal preparation.
Includes pharmacies, supply rooms, charging stations, congestion zones,
patient bed with approach preferences, and kitchen area (pantry, prep station, stove).

MODIFIED: Stock system uses integer counts per location (not booleans)
"""

import numpy as np
import mujoco
import mujoco.viewer
from typing import Dict, Optional, Tuple


class ExpandedHospitalMuJoCoEnv:
    """
    Expanded 6D MuJoCo environment with PDDL-aligned locations and dynamic state.
    """

    def __init__(self, render_mode: str = "human"):
        self.render_mode = render_mode
        self.model = self._create_expanded_hospital_model()
        self.data = mujoco.MjData(self.model)

        # MPC-compatible timing
        self.dt = 0.2  # Main control timestep (matches MPC Ts)
        self.mujoco_dt = 0.01  # Physics timestep
        self.steps_per_control = int(self.dt / self.mujoco_dt)

        # Expanded hospital ward with kitchen area
        self.locations = {
            # Starting area
            "home": np.array([0.0, 0.0]),
            # Pharmacies (medication sources)
            "pharmacy_north": np.array([5.0, 18.0]),
            "pharmacy_south": np.array([6.0, -15.0]),
            # Supply rooms
            "supply_A": np.array([14.0, 10.0]),
            "supply_B": np.array([15.0, -12.0]),
            # Charging stations
            "charge_main": np.array([3.0, 5.0]),
            "charge_backup": np.array([17.0, -18.0]),
            # Patient bed approach locations
            "patient_bed_left": np.array([20.5, 12.0]),
            "patient_bed_right": np.array([23.5, 10.0]),
            # Kitchen area (north-west quadrant)
            "pantry": np.array([-3.0, 15.0]),
            "fridge": np.array([-6.0, 17.5]),
            "prep_station": np.array([0.0, 20.0]),
            "stove": np.array([-3.0, 20.5]),
            "quality_check": np.array([3.0, 21.0]),
        }

        # Location metadata for planning
        self.location_metadata = {
            "pharmacy_north": {
                "type": "pharmacy",
                "congestion": 0.2,
                "size": 1.0,
                "approach_radius": 1.5,
                "initial_stock": 5,
            },
            "pharmacy_south": {
                "type": "pharmacy",
                "congestion": 0.1,
                "size": 1.0,
                "approach_radius": 1.5,
                "initial_stock": 2,
            },
            "supply_A": {
                "type": "supply",
                "stock_probability": 0.7,
                "size": 0.8,
                "approach_radius": 1.2,
                "initial_stock": 7,
            },
            "supply_B": {
                "type": "supply",
                "stock_probability": 0.9,
                "size": 0.8,
                "approach_radius": 1.2,
                "initial_stock": 1,
            },
            "charge_main": {
                "type": "charger",
                "charge_rate": 1.0,
                "size": 0.6,
                "approach_radius": 1.0,
            },
            "charge_backup": {
                "type": "charger",
                "charge_rate": 1.0,
                "size": 0.6,
                "approach_radius": 1.0,
            },
            "patient_bed_left": {
                "type": "goal",
                "approach_sides": ["left"],
                "approach_tolerance": 0.8,
                "size": 1.0,
            },
            "patient_bed_right": {
                "type": "goal",
                "approach_sides": ["right"],
                "approach_tolerance": 0.8,
                "size": 1.0,
            },
            # Kitchen area
            "pantry": {
                "type": "food_storage",
                "congestion": 0.15,
                "size": 1.0,
                "approach_radius": 1.2,
                # Per-ingredient stock (PDDL plus legacy meal keys).
                "initial_stock": {
                    "bread": 10,
                    "nuts": 10,
                    "vegetables": 10,
                    "sandwich": 10,
                    "soup": 5,
                    "full_meal": 3,
                },
            },
            "fridge": {
                "type": "food_storage",
                "congestion": 0.17,
                "size": 1.0,
                "approach_radius": 1.2,
                "initial_stock": {
                    "chicken": 5,
                },
            },
            "prep_station": {
                "type": "kitchen",
                "congestion": 0.2,
                "size": 0.8,
                "approach_radius": 1.0,
            },
            "stove": {
                "type": "kitchen",
                "congestion": 0.25,
                "size": 0.6,
                "approach_radius": 0.8,
                "hazard": "heat",
            },
            "quality_check": {
                "type": "quality_check",
                "congestion": 0.1,
                "size": 0.8,
                "approach_radius": 1.0,
            },
        }

        # Dynamic environment state (changes during episodes)
        self.environment_state = {
            # Backward-compatible boolean accessors (derived from counts)
            "supply_A_in_stock": True,
            "supply_B_in_stock": True,
            # Dynamic conditions
            "battery_level": 1.0,  # Start at 100%
        }
        # Stock levels initialized from location_metadata (all locations)
        self._init_stock_levels()

        # Environment event tracking
        self.environment_events = []

        # State tracking
        self.robot_state_6d = np.zeros(6)
        self.step_count = 0
        self.episode_count = 0
        self.last_control = np.zeros(3)
        self.previous_position = None

        # Control limits (matches MPC constraints)
        self.max_acceleration = 2.0
        self.max_angular_acceleration = 3.0

        # Viewer
        self.viewer = None
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        print("Expanded PDDL-Aligned Hospital Environment Ready")
        print(f"  Floor size: 50x50 meters")
        print(f"  Locations: {len(self.locations)}")
        print(f"  State space: [x, y, θ, vx, vy, ωz]")
        print(f"  Control space: [ax, ay, α]")
        stock_summary = ", ".join(
            f"{k}={v}"
            for k, v in self.environment_state.items()
            if k.endswith("_stock")
        )
        print(f"  Stock levels: {stock_summary}")

    def _create_expanded_hospital_model(self):
        """Create expanded 50x50m hospital ward with PDDL-aligned locations."""

        xml_string = """
        <mujoco model="expanded_hospital_ward_6d">
            <compiler angle="radian"/>
            <option timestep="0.01" integrator="RK4"/>
            <default>
                <geom contype="0" conaffinity="0"/>
            </default>
            
            <asset>
                <!-- Materials -->
                <material name="robot" rgba="0.2 0.5 0.9 1"/>
                <material name="pharmacy" rgba="0.1 0.7 0.3 1"/>
                <material name="supply" rgba="0.9 0.6 0.2 1"/>
                <material name="charger" rgba="0.9 0.9 0.1 1"/>
                <material name="bed" rgba="0.98 0.94 0.9 1"/>
                <material name="home_zone" rgba="0.2 0.6 1.0 0.4"/>
                <material name="congestion_zone" rgba="1.0 0.3 0.3 0.3"/>
                <material name="narrow_zone" rgba="0.8 0.5 0.0 0.3"/>
                <material name="approach_left" rgba="0.3 0.9 0.3 0.3"/>
                <material name="approach_right" rgba="0.3 0.3 0.9 0.3"/>
                <material name="kitchen" rgba="0.95 0.85 0.7 1"/>
                <material name="stove_hot" rgba="0.9 0.2 0.1 0.8"/>
                
                <!-- Hospital Floor Pattern -->
                <texture name="hospital_floor" type="2d" builtin="checker" 
                         width="100" height="100" 
                         rgb1="0.92 0.94 0.96" rgb2="0.88 0.90 0.94"/>
                <material name="floor" texture="hospital_floor" rgba="1 1 1 1"/>
            </asset>
            
            <worldbody>
                <!-- LARGE HOSPITAL WARD FLOOR: 50x50 meters -->
                <geom name="floor" type="plane" size="50 50 0.1" material="floor"/>
                
                <!-- PERIMETER WALLS -->
                <geom name="wall_north" type="box" size="25 0.2 2.0" 
                      pos="0 25 2.0" rgba="0.85 0.88 0.92 0.6"/>
                <geom name="wall_south" type="box" size="25 0.2 2.0" 
                      pos="0 -25 2.0" rgba="0.85 0.88 0.92 0.6"/>
                <geom name="wall_east" type="box" size="0.2 25 2.0" 
                      pos="25 0 2.0" rgba="0.85 0.88 0.92 0.6"/>
                <geom name="wall_west" type="box" size="0.2 25 2.0" 
                      pos="-25 0 2.0" rgba="0.85 0.88 0.92 0.6"/>
                
                <!-- LIGHTING -->
                <light name="main_light_1" pos="10 10 15" diffuse="0.9 0.9 1.0"/>
                <light name="main_light_2" pos="20 -10 15" diffuse="0.9 0.9 1.0"/>
                <light name="main_light_3" pos="-10 10 15" diffuse="0.9 0.9 1.0"/>
                <light name="ambient" pos="0 0 20" diffuse="0.3 0.3 0.4"/>
                
                <!-- LOCATION 1: HOME BASE (0, 0) -->
                <geom name="home_marker" type="cylinder" size="0.8 0.05" 
                      pos="0 0 0.05" material="home_zone"/>
                <geom name="home_center" type="sphere" size="0.3" 
                      pos="0 0 0.3" rgba="0.2 0.6 1.0 0.6"/>
                
                <!-- LOCATION 2: PHARMACY NORTH (5, 18) -->
                <geom name="pharmacy_north_base" type="box" size="1.0 0.8 0.05" 
                      pos="5 18 0.05" rgba="0.1 0.9 0.3 0.4"/>
                <geom name="pharmacy_north_cabinet" type="box" size="0.9 0.7 1.0" 
                      pos="5 18 1.0" material="pharmacy"/>
                <geom name="pharmacy_north_counter" type="box" size="0.95 0.75 0.05" 
                      pos="5 18 2.05" rgba="0.7 0.9 0.7 1"/>
                <geom name="pharmacy_north_window" type="box" size="0.4 0.05 0.3" 
                      pos="5 18.7 1.3" rgba="0.2 0.2 0.2 0.8"/>
                <geom name="pharmacy_north_dispenser" type="cylinder" size="0.15 0.2" 
                      pos="5.5 18.3 1.2" rgba="0.2 0.9 0.4 1"/>
                
                <!-- LOCATION 3: PHARMACY SOUTH (6, -15) -->
                <geom name="pharmacy_south_base" type="box" size="1.0 0.8 0.05" 
                      pos="6 -15 0.05" rgba="0.1 0.9 0.3 0.4"/>
                <geom name="pharmacy_south_cabinet" type="box" size="0.9 0.7 1.0" 
                      pos="6 -15 1.0" material="pharmacy"/>
                <geom name="pharmacy_south_counter" type="box" size="0.95 0.75 0.05" 
                      pos="6 -15 2.05" rgba="0.7 0.9 0.7 1"/>
                <geom name="pharmacy_south_window" type="box" size="0.4 0.05 0.3" 
                      pos="6 -14.3 1.3" rgba="0.2 0.2 0.2 0.8"/>
                <geom name="pharmacy_south_dispenser" type="cylinder" size="0.15 0.2" 
                      pos="6.5 -14.7 1.2" rgba="0.2 0.9 0.4 1"/>
                
                <!-- LOCATION 4: SUPPLY ROOM A (14, 10) -->
                <geom name="supply_A_base" type="box" size="0.8 0.6 0.05" 
                      pos="14 10 0.05" rgba="0.9 0.6 0.2 0.4"/>
                <geom name="supply_A_cabinet" type="box" size="0.7 0.5 0.9" 
                      pos="14 10 0.9" material="supply"/>
                <geom name="supply_A_shelf_1" type="box" size="0.65 0.45 0.05" 
                      pos="14 10 0.6" rgba="0.8 0.8 0.9 1"/>
                <geom name="supply_A_shelf_2" type="box" size="0.65 0.45 0.05" 
                      pos="14 10 1.2" rgba="0.8 0.8 0.9 1"/>
                <geom name="supply_A_items" type="box" size="0.2 0.15 0.1" 
                      pos="14 10 1.3" rgba="0.9 0.5 0.1 1"/>
                
                <!-- LOCATION 5: SUPPLY ROOM B (15, -12) -->
                <geom name="supply_B_base" type="box" size="0.8 0.6 0.05" 
                      pos="15 -12 0.05" rgba="0.9 0.6 0.2 0.4"/>
                <geom name="supply_B_cabinet" type="box" size="0.7 0.5 0.9" 
                      pos="15 -12 0.9" material="supply"/>
                <geom name="supply_B_shelf_1" type="box" size="0.65 0.45 0.05" 
                      pos="15 -12 0.6" rgba="0.8 0.8 0.9 1"/>
                <geom name="supply_B_shelf_2" type="box" size="0.65 0.45 0.05" 
                      pos="15 -12 1.2" rgba="0.8 0.8 0.9 1"/>
                <geom name="supply_B_items" type="box" size="0.2 0.15 0.1" 
                      pos="15 -12 1.3" rgba="0.9 0.5 0.1 1"/>
                
                <!-- LOCATION 6: CHARGING STATION MAIN (3, 5) -->
                <geom name="charge_main_base" type="box" size="0.6 0.6 0.05" 
                      pos="3 5 0.05" rgba="0.9 0.9 0.1 0.4"/>
                <geom name="charge_main_station" type="cylinder" size="0.5 0.5" 
                      pos="3 5 0.5" material="charger"/>
                <geom name="charge_main_plug" type="box" size="0.15 0.15 0.4" 
                      pos="3 5.5 0.6" rgba="0.3 0.3 0.3 1"/>
                <geom name="charge_main_indicator" type="sphere" size="0.1" 
                      pos="3 5 1.1" rgba="0.0 1.0 0.0 1"/>
                
                <!-- LOCATION 7: CHARGING STATION BACKUP (17, -18) -->
                <geom name="charge_backup_base" type="box" size="0.6 0.6 0.05" 
                      pos="17 -18 0.05" rgba="0.9 0.9 0.1 0.4"/>
                <geom name="charge_backup_station" type="cylinder" size="0.5 0.5" 
                      pos="17 -18 0.5" material="charger"/>
                <geom name="charge_backup_plug" type="box" size="0.15 0.15 0.4" 
                      pos="17 -17.5 0.6" rgba="0.3 0.3 0.3 1"/>
                <geom name="charge_backup_indicator" type="sphere" size="0.1" 
                      pos="17 -18 1.1" rgba="0.0 1.0 0.0 1"/>
                
                <!-- PATIENT BED PHYSICAL GEOMETRY -->
                <geom name="patient_bed_base" type="box" size="1.2 2.2 0.5" 
                      pos="22 10 0.5" material="bed"/>
                <geom name="patient_bed_mattress" type="box" size="1.1 2.0 0.15" 
                      pos="22 10 1.15" rgba="0.95 0.90 0.85 1"/>
                <geom name="patient_pillow" type="box" size="0.4 0.6 0.1" 
                      pos="22 11.2 1.3" rgba="1.0 1.0 1.0 1"/>
                <geom name="patient_blanket" type="box" size="1.0 1.5 0.05" 
                      pos="22 9.5 1.25" rgba="0.7 0.85 0.95 0.9"/>
                <geom name="bed_frame_head" type="box" size="1.3 0.1 0.6" 
                      pos="22 11.5 0.8" rgba="0.8 0.8 0.85 1"/>
                <geom name="bed_frame_foot" type="box" size="1.3 0.1 0.4" 
                      pos="22 8.5 0.6" rgba="0.8 0.8 0.85 1"/>
                <geom name="bed_left_approach_zone" type="box" size="0.8 2.2 0.03" 
                      pos="20.5 10 0.03" material="approach_left" contype="0" conaffinity="0"/>
                <geom name="bed_left_marker" type="cylinder" size="0.3 0.05" 
                      pos="20.5 10 0.05" rgba="0.3 0.9 0.3 0.5"/>
                <geom name="bed_right_approach_zone" type="box" size="0.8 2.2 0.03" 
                      pos="23.5 10 0.03" material="approach_right" contype="0" conaffinity="0"/>
                <geom name="bed_right_marker" type="cylinder" size="0.3 0.05" 
                      pos="23.5 12 0.05" rgba="0.3 0.3 0.9 0.5"/>
                
                <!-- LOCATION 11: PANTRY (-3, 15) - Ingredient storage -->
                <geom name="pantry_base" type="box" size="1.0 0.8 0.05" 
                      pos="-3 15 0.05" rgba="0.95 0.85 0.7 0.4"/>
                <geom name="pantry_shelves" type="box" size="0.9 0.7 1.2" 
                      pos="-3 15 1.2" material="kitchen"/>
                <geom name="pantry_shelf_1" type="box" size="0.85 0.65 0.05" 
                      pos="-3 15 0.7" rgba="0.85 0.75 0.6 1"/>
                <geom name="pantry_shelf_2" type="box" size="0.85 0.65 0.05" 
                      pos="-3 15 1.4" rgba="0.85 0.75 0.6 1"/>
                <geom name="pantry_shelf_3" type="box" size="0.85 0.65 0.05" 
                      pos="-3 15 2.1" rgba="0.85 0.75 0.6 1"/>
                <geom name="pantry_items_1" type="box" size="0.15 0.12 0.1" 
                      pos="-3.3 15.2 0.85" rgba="0.9 0.4 0.2 1"/>
                <geom name="pantry_items_2" type="sphere" size="0.1" 
                      pos="-2.7 14.8 1.55" rgba="0.2 0.8 0.2 1"/>
                
                <!-- LOCATION 12: PREP STATION (0, 20) - Chopping/assembly/plating -->
                <geom name="prep_base" type="box" size="0.8 0.6 0.05" 
                      pos="0 20 0.05" rgba="0.9 0.9 0.95 0.4"/>
                <geom name="prep_counter" type="box" size="0.8 0.6 0.9" 
                      pos="0 20 0.9" rgba="0.85 0.85 0.9 1"/>
                <geom name="prep_surface" type="box" size="0.85 0.65 0.05" 
                      pos="0 20 1.85" rgba="0.95 0.95 0.98 1"/>
                <geom name="prep_cutting_board" type="box" size="0.25 0.18 0.02" 
                      pos="0.2 20 1.9" rgba="0.75 0.6 0.4 1"/>
                <geom name="prep_bowl" type="cylinder" size="0.12 0.08" 
                      pos="-0.3 20.2 1.95" rgba="0.9 0.9 0.95 1"/>
                
                <!-- LOCATION 13: STOVE (-3, 20.5) - Cooking station -->
                <geom name="stove_base" type="box" size="0.6 0.6 0.05"
                      pos="-3 20.5 0.05" rgba="0.3 0.3 0.35 0.4"/>
                <geom name="stove_body" type="box" size="0.6 0.6 0.8"
                      pos="-3 20.5 0.8" rgba="0.35 0.35 0.4 1"/>
                <geom name="stove_top" type="box" size="0.65 0.65 0.05"
                      pos="-3 20.5 1.65" rgba="0.2 0.2 0.25 1"/>
                <geom name="stove_burner_1" type="cylinder" size="0.15 0.02"
                      pos="-3.2 20.7 1.7" material="stove_hot"/>
                <geom name="stove_burner_2" type="cylinder" size="0.15 0.02"
                      pos="-2.8 20.3 1.7" material="stove_hot"/>
                <geom name="stove_pot" type="cylinder" size="0.13 0.15"
                      pos="-3.2 20.7 1.87" rgba="0.6 0.6 0.65 1"/>
                <geom name="stove_heat_zone" type="cylinder" size="0.8 0.03"
                      pos="-3 20.5 0.03" rgba="1.0 0.3 0.1 0.2" contype="0" conaffinity="0"/>
                
                <!-- ROBOT (6D Mobile Base) -->
                <body name="robot" pos="0 0 0.3">
                    <geom name="robot_body" type="cylinder" size="0.25 0.1" 
                          material="robot" mass="3"/>
                    <geom name="robot_arm_base" type="cylinder" size="0.03 0.05" 
                          pos="0.15 0 0.05" rgba="0.9 0.5 0.1 1"/>
                    <geom name="robot_arm_extend" type="box" size="0.2 0.02 0.02" 
                          pos="0.35 0 0.05" rgba="1.0 0.4 0.0 1"/>
                    <geom name="robot_arm_tip" type="sphere" size="0.03" 
                          pos="0.55 0 0.05" rgba="1.0 0.2 0.0 1"/>
                    <geom name="front_marker" type="box" size="0.05 0.05 0.02" 
                          pos="0.3 0 0.12" rgba="0.2 0.9 0.3 1"/>
                    <joint name="slide_x" type="slide" axis="1 0 0" damping="0.3"/>
                    <joint name="slide_y" type="slide" axis="0 1 0" damping="0.3"/>
                    <joint name="hinge_z" type="hinge" axis="0 0 1" damping="0.1"/>
                </body>
            </worldbody>
            
            <actuator>
                <motor name="force_x" joint="slide_x" gear="1"/>
                <motor name="force_y" joint="slide_y" gear="1"/>
                <motor name="torque_z" joint="hinge_z" gear="1"/>
            </actuator>
        </mujoco>
        """

        return mujoco.MjModel.from_xml_string(xml_string)

    # -----------------------------------------------------------------
    # Stock Management (NEW)
    # -----------------------------------------------------------------

    def _init_stock_levels(self):
        """Initialize stock counts from location_metadata."""
        for loc_name, meta in self.location_metadata.items():
            if "initial_stock" not in meta:
                continue
            stock = meta["initial_stock"]
            if isinstance(stock, dict):
                # Per-item stock (e.g. pantry with sandwich/soup/full_meal)
                for item_name, count in stock.items():
                    stock_key = f"{loc_name}_{item_name}_stock"
                    self.environment_state[stock_key] = count
            else:
                # Single stock count (e.g. pharmacy_north)
                stock_key = f"{loc_name}_stock"
                self.environment_state[stock_key] = stock

        # Sync backward-compatible boolean accessors
        self._sync_stock_booleans()

    def _sync_stock_booleans(self):
        """Keep boolean in_stock flags in sync with counts for backward compat."""
        self.environment_state["supply_A_in_stock"] = (
            self.environment_state.get("supply_A_stock", 0) > 0
        )
        self.environment_state["supply_B_in_stock"] = (
            self.environment_state.get("supply_B_stock", 0) > 0
        )

    def get_stock(self, location: str) -> int:
        """
        Get current stock count at a location.

        Returns 0 for locations without stock (chargers, passthrough, etc).
        """
        stock_key = f"{location}_stock"
        return self.environment_state.get(stock_key, 0)

    def consume_stock(self, location: str, amount: int = 1) -> bool:
        """
        Consume stock at a location. Returns True if successful.

        Decrements count and fires stock_low/stock_out events.
        """
        stock_key = f"{location}_stock"
        current = self.environment_state.get(stock_key, 0)

        if current < amount:
            self.environment_events.append(
                {
                    "type": "stock_out",
                    "location": location,
                    "requested": amount,
                    "available": current,
                    "time": self.step_count * self.dt,
                }
            )
            return False

        self.environment_state[stock_key] = current - amount

        # Fire low-stock warning at 1 remaining
        if self.environment_state[stock_key] == 1:
            self.environment_events.append(
                {
                    "type": "stock_low",
                    "location": location,
                    "remaining": 1,
                    "time": self.step_count * self.dt,
                }
            )
        elif self.environment_state[stock_key] == 0:
            self.environment_events.append(
                {
                    "type": "stock_depleted",
                    "location": location,
                    "time": self.step_count * self.dt,
                }
            )

        self._sync_stock_booleans()
        return True

    def get_all_stock_levels(self) -> Dict[str, int]:
        """Get stock counts for all stocked locations/items."""
        result = {}
        for loc, meta in self.location_metadata.items():
            if "initial_stock" not in meta:
                continue
            stock = meta["initial_stock"]
            if isinstance(stock, dict):
                # Per-item stock (e.g. pantry_sandwich, pantry_soup)
                for item_name in stock:
                    key = f"{loc}_{item_name}"
                    result[key] = self.get_stock(key)
            else:
                result[loc] = self.get_stock(loc)
        return result

    # -----------------------------------------------------------------
    # Core Environment Methods
    # -----------------------------------------------------------------

    def reset(
        self,
        initial_position: Optional[np.ndarray] = None,
        initial_orientation: float = 0.0,
    ) -> np.ndarray:
        """Reset environment to initial state."""

        self.episode_count += 1
        self.step_count = 0

        if initial_position is None:
            initial_position = self.locations["charge_main"].copy()

        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)

        # Set initial conditions
        self.data.qpos[0] = initial_position[0]
        self.data.qpos[1] = initial_position[1]
        self.data.qpos[2] = initial_orientation
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self.robot_state_6d = self._get_robot_state_6d()

        # Reset environment state (stock initialized from metadata)
        self.environment_state = {
            "supply_A_in_stock": True,
            "supply_B_in_stock": True,
            "battery_level": 1.0,
        }
        self._init_stock_levels()
        self.environment_events = []
        self.previous_position = None

        mujoco.mj_forward(self.model, self.data)
        self.robot_state_6d = self._get_robot_state_6d()

        print(f"\n=== Episode {self.episode_count} Started ===")
        print(
            f"Initial position: [{self.robot_state_6d[0]:.2f}, {self.robot_state_6d[1]:.2f}]"
        )
        print(
            f"Expected position: [{initial_position[0]:.2f}, {initial_position[1]:.2f}]"
        )
        print(
            f"At location: {self._get_current_location_name(self.robot_state_6d[:2])}"
        )
        print(f"Initial state: {self.robot_state_6d}")

        return self.robot_state_6d.copy()

    def set_robot_pose(
        self,
        position: np.ndarray,
        orientation: float = 0.0,
        zero_velocity: bool = True,
    ) -> np.ndarray:
        """Set the robot pose and keep MuJoCo/data state synchronized."""
        pos = np.array(position, dtype=float)
        self.data.qpos[0] = pos[0]
        self.data.qpos[1] = pos[1]
        self.data.qpos[2] = float(orientation)
        if zero_velocity:
            self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.robot_state_6d = self._get_robot_state_6d()
        self.previous_position = self.robot_state_6d[:2].copy()
        return self.robot_state_6d.copy()

    def step(self, control: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Execute one control step with expanded environment feedback."""

        self.step_count += 1

        # Validate and clamp control
        if not isinstance(control, np.ndarray) or len(control) != 3:
            raise ValueError(f"Control must be 3D [ax, ay, α], got {type(control)}")

        ax = np.clip(control[0], -self.max_acceleration, self.max_acceleration)
        ay = np.clip(control[1], -self.max_acceleration, self.max_acceleration)
        alpha = np.clip(
            control[2], -self.max_angular_acceleration, self.max_angular_acceleration
        )

        clamped_control = np.array([ax, ay, alpha])
        self.last_control = clamped_control.copy()

        # Convert to forces/torques
        robot_mass = 3.0
        robot_inertia = 1.0

        force_x = ax * robot_mass
        force_y = ay * robot_mass
        torque_z = alpha * robot_inertia

        # Execute physics
        for _ in range(self.steps_per_control):
            self.data.ctrl[0] = force_x
            self.data.ctrl[1] = force_y
            self.data.ctrl[2] = torque_z

            mujoco.mj_step(self.model, self.data)

            if self.viewer is not None:
                self.viewer.sync()

        # Get new state
        new_6d_state = self._get_robot_state_6d()
        self.robot_state_6d = new_6d_state

        # Calculate distance traveled and update battery
        if self.previous_position is not None:
            distance_traveled = np.linalg.norm(
                new_6d_state[:2] - self.previous_position
            )
            self.update_battery(distance_traveled)
        else:
            distance_traveled = 0

        self.previous_position = new_6d_state[:2].copy()

        # Determine current location
        current_loc = self._get_current_location_name(new_6d_state[:2])

        # Comprehensive step info
        step_info = {
            "step_count": self.step_count,
            "simulation_time": self.step_count * self.dt,
            "control_applied": clamped_control.copy(),
            "control_clamped": not np.allclose(control, clamped_control, atol=1e-6),
            "position": new_6d_state[:2].copy(),
            "orientation": new_6d_state[2],
            "linear_velocity": new_6d_state[3:5].copy(),
            "angular_velocity": new_6d_state[5],
            "speed": np.linalg.norm(new_6d_state[3:5]),
            "distance_traveled": distance_traveled,
            "battery_level": self.environment_state["battery_level"],
            "battery_critical": self.environment_state["battery_level"] < 0.15,
            "current_location": current_loc,
            "at_location": current_loc if current_loc != "traveling" else None,
            "congestion_penalty": self.get_congestion_penalty(current_loc),
            "approach_side": (
                self._detect_approach_side(new_6d_state[:2])
                if "patient_bed" in current_loc
                else None
            ),
            "location_distances": {
                name: np.linalg.norm(new_6d_state[:2] - pos)
                for name, pos in self.locations.items()
            },
            "environment_events": self.environment_events.copy(),
        }

        # Clear events after reporting
        self.environment_events = []

        return new_6d_state.copy(), step_info

    def _get_robot_state_6d(self) -> np.ndarray:
        """Extract 6D state: [x, y, θ, vx, vy, ωz]"""
        x = self.data.qpos[0]
        y = self.data.qpos[1]
        theta = self.data.qpos[2]
        theta = np.arctan2(np.sin(theta), np.cos(theta))

        vx = self.data.qvel[0]
        vy = self.data.qvel[1]
        omega_z = self.data.qvel[2]

        return np.array([x, y, theta, vx, vy, omega_z])

    def set_robot_state_6d(self, state_6d: np.ndarray) -> np.ndarray:
        """Set the MuJoCo robot state and keep cached state in sync."""
        state = np.array(state_6d, dtype=float).reshape(6)
        self.data.qpos[0] = state[0]
        self.data.qpos[1] = state[1]
        self.data.qpos[2] = state[2]
        self.data.qvel[0] = state[3]
        self.data.qvel[1] = state[4]
        self.data.qvel[2] = state[5]
        self.data.ctrl[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self.robot_state_6d = self._get_robot_state_6d()
        self.previous_position = self.robot_state_6d[:2].copy()
        return self.robot_state_6d.copy()

    def _get_current_location_name(
        self, position: np.ndarray, tolerance: float = 1.5
    ) -> str:
        """Determine current location from position."""
        for location_name, location_pos in self.locations.items():
            distance = np.linalg.norm(position - location_pos)
            if distance < tolerance:
                return location_name
        return "traveling"

    def _detect_approach_side(self, position: np.ndarray) -> str:
        """Detect approach side for patient bed."""
        left = self.locations["patient_bed_left"]
        right = self.locations["patient_bed_right"]
        bed_pos = 0.5 * (left + right)
        relative_x = position[0] - bed_pos[0]

        if abs(relative_x) < 0.5:
            return "center"
        elif relative_x < 0:
            return "left"
        else:
            return "right"

    def simulate_stock_check(self, location: str) -> bool:
        """
        Check if location has stock available.

        Now uses integer counts instead of random probability.
        Returns True if stock > 0.
        """
        stock = self.get_stock(location)
        if stock <= 0:
            self.environment_events.append(
                {
                    "type": "stock_out",
                    "location": location,
                    "time": self.step_count * self.dt,
                }
            )
            return False
        return True

    def update_battery(self, distance_traveled: float):
        """Deplete battery based on distance (10% per unit)."""
        battery_cost = distance_traveled * 0.01
        old_level = self.environment_state["battery_level"]
        self.environment_state["battery_level"] -= battery_cost
        self.environment_state["battery_level"] = max(
            0, self.environment_state["battery_level"]
        )

        if self.environment_state["battery_level"] < 0.15 and old_level >= 0.15:
            self.environment_events.append(
                {
                    "type": "battery_critical",
                    "level": self.environment_state["battery_level"],
                    "time": self.step_count * self.dt,
                }
            )

    def recharge_battery(self):
        """Recharge to 100%."""
        old_level = self.environment_state["battery_level"]
        self.environment_state["battery_level"] = 1.0
        self.environment_events.append(
            {
                "type": "recharge",
                "from": old_level,
                "to": 1.0,
                "time": self.step_count * self.dt,
            }
        )

    def get_congestion_penalty(self, location: str) -> float:
        """Get time penalty multiplier for congested zones."""
        if location in self.location_metadata:
            return 1.0 + self.location_metadata[location].get("congestion", 0.0)
        return 1.0

    def get_location_distance(self, location_name: str) -> float:
        """Get distance to named location."""
        if location_name not in self.locations:
            return float("inf")
        target_pos = self.locations[location_name]
        current_pos = self.robot_state_6d[:2]
        return np.linalg.norm(current_pos - target_pos)

    def is_at_location(self, location_name: str, tolerance: float = 1.0) -> bool:
        """Check if robot is at specified location."""
        distance = self.get_location_distance(location_name)
        return distance < tolerance

    def get_simulation_state(self) -> Dict:
        """Get complete simulation state."""
        return {
            "robot_6d_state": self.robot_state_6d.copy(),
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "simulation_time": self.step_count * self.dt,
            "battery_level": self.environment_state["battery_level"],
            "environment_state": self.environment_state.copy(),
            "stock_levels": self.get_all_stock_levels(),
            "current_location": self._get_current_location_name(
                self.robot_state_6d[:2]
            ),
            "location_distances": {
                name: self.get_location_distance(name) for name in self.locations.keys()
            },
        }

    def render(self):
        """Render (handled by viewer.sync())."""
        pass

    def close(self):
        """Clean shutdown."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        print("Expanded Hospital Environment closed")


def test_expanded_environment():
    """Test the expanded 10-location environment."""

    print("Testing Expanded Hospital Environment")
    print("=" * 60)

    env = ExpandedHospitalMuJoCoEnv(render_mode="human")

    # Reset
    initial_state = env.reset()
    print(f"\nInitial state: {initial_state}")

    # Print stock levels
    print("\nStock levels:")
    for loc, count in env.get_all_stock_levels().items():
        print(f"  {loc:20s}: {count} units")

    # Test stock consumption
    print("\nTesting stock consumption:")
    print(f"  supply_A stock: {env.get_stock('supply_A')}")
    ok = env.consume_stock("supply_A")
    print(f"  Consumed from supply_A: {ok}, remaining: {env.get_stock('supply_A')}")

    print(f"  supply_B stock: {env.get_stock('supply_B')}")
    ok = env.consume_stock("supply_B")
    print(f"  Consumed from supply_B: {ok}, remaining: {env.get_stock('supply_B')}")
    ok2 = env.consume_stock("supply_B")
    print(f"  Second consume from supply_B: {ok2} (should be False)")

    # Print all location distances
    print("\nDistances from home to all locations:")
    for name, pos in env.locations.items():
        dist = np.linalg.norm(pos - initial_state[:2])
        print(f"  {name:20s}: {dist:6.2f}m")

    print("\nPress Enter to close...")
    input()
    env.close()


if __name__ == "__main__":
    test_expanded_environment()
