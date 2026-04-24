#!/usr/bin/env python3
"""
Enhanced Medication Reward Engine with Preference Feature Tracking
==================================================================

Reward engine for medication delivery with integrated preference learning.
Tracks episode features (time, safety, battery, proximity, approach) for
patient preference learning via projected gradient descent.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from enum import Enum

# Import shared enums for consistency
from multilevel_control_stack.shared_actions.medicatePatient import MedicationAction


class MedicationRewardEngine:
    """
    Enhanced reward engine with preference feature tracking.
    
    Evaluates task completion, safety, efficiency AND extracts features
    for preference learning across 5 dimensions:
    - Time: Episode duration
    - Safety: Minimum safety margin to patient
    - Battery: Total energy consumption
    - Proximity: Movement comfort near patient
    - Approach: Final positioning quality
    """
    
    def __init__(self, locations: Dict[str, np.ndarray], 
                 goal_tolerance: float = 0.6,
                 time_penalty: float = -0.005,
                 success_bonus: float = 15.0,
                 efficiency_bonus: float = 8.0):
        """
        Initialize medication reward engine with preference tracking.
        
        Args:
            locations: Dictionary of location names to positions
            goal_tolerance: Distance tolerance for task completion
            time_penalty: Penalty per time unit
            success_bonus: Bonus for completing tasks
            efficiency_bonus: Bonus for efficient execution
        """
        
        self.locations = locations
        self.goal_tolerance = goal_tolerance
        self.time_penalty = time_penalty
        self.success_bonus = success_bonus
        self.efficiency_bonus = efficiency_bonus
        
        # Task state tracking
        self.medication_retrieved = False
        self.medication_delivered = False
        self.robot_at_home = False
        
        # Episode tracking (existing)
        self.episode_rewards = []
        self.episode_safety_violations = 0
        self.episode_tasks_completed = 0
        
        # Safety parameters
        self.patient_safety_radius = 2.0
        self.min_safe_velocity = 0.1
        self.max_safe_velocity = 1.0
        
        # NEW: Preference feature tracking
        self.episode_features = {
            'total_time': 0.0,
            'min_safety_margin': float('inf'),  # Track minimum distance to patient
            'total_battery_used': 0.0,
            'proximity_comfort_scores': [],     # Comfort when near patient
            'approach_quality_scores': [],      # Positioning accuracy
            'velocity_profile': [],             # Velocity history for analysis
            'patient_proximity_events': 0       # Times robot was near patient
        }
        
        # Feature normalization constants (tuned for hospital environment)
        self.NORM_TIME_MAX = 120.0      # Max expected episode time (seconds)
        self.NORM_SAFETY_MIN = 0.5      # Min acceptable safety margin (meters)
        self.NORM_SAFETY_MAX = 3.0      # Max safety margin we care about
        self.NORM_BATTERY_MAX = 100.0   # Max battery units per episode
        
        print("Enhanced MedicationRewardEngine initialized")
        print(f"  Locations: {list(locations.keys())}")
        print(f"  Goal tolerance: {goal_tolerance}m")
        print(f"  Patient safety zone: {self.patient_safety_radius}m radius")
        print(f"  ✨ Preference feature tracking enabled")
    
    def evaluate_delivery_step(self, prev_state: np.ndarray, 
                              next_state: np.ndarray,
                              task: str,
                              target_state: np.ndarray,
                              execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a medication delivery step AND accumulate preference features.
        
        Args:
            prev_state: 6D state before action [x, y, θ, vx, vy, ωz]
            next_state: 6D state after action
            task: Task identifier (MedicationAction enum or string)
            target_state: Intended target state
            execution_data: MPC execution results
            
        Returns:
            Comprehensive reward feedback including feature accumulation
        """
        
        # Extract positions and velocities
        prev_pos = prev_state[:2]
        next_pos = next_state[:2]
        next_velocity = np.linalg.norm(next_state[3:5])
        
        # Handle MedicationAction enum or string input
        if hasattr(task, 'value'):
            task_name = task.value
        else:
            task_name = str(task)
            
        print(f"  Evaluating: {task_name}")
        
        # =====================================================================
        # REWARD CALCULATION (Existing Logic)
        # =====================================================================
        
        reward = 0.0
        progress = 0.0
        task_completed = False
        
        # Task completion logic (using your existing logic)
        if 'move_to_med_station' in task_name.lower():
            if 'med_station' in self.locations:
                target_pos = self.locations['med_station']
                distance = np.linalg.norm(next_pos - target_pos)
                prev_distance = np.linalg.norm(prev_pos - target_pos)
                progress = prev_distance - distance
                
                if distance < self.goal_tolerance:
                    task_completed = True
                    reward += self.success_bonus
                    print(f"    ✓ Reached medication station!")
        
        elif 'retrieve_medication' in task_name.lower():
            if 'med_station' in self.locations:
                med_pos = self.locations['med_station']
                distance = np.linalg.norm(next_pos - med_pos)
                
                if distance < self.goal_tolerance:
                    task_completed = True
                    reward += self.success_bonus
                    self.medication_retrieved = True
                    print(f"    ✓ Medication retrieved!")
        
        elif 'move_to_patient' in task_name.lower():
            if 'patient_bed' in self.locations:
                target_pos = self.locations['patient_bed']
                distance = np.linalg.norm(next_pos - target_pos)
                prev_distance = np.linalg.norm(prev_pos - target_pos)
                progress = prev_distance - distance
                
                if distance < self.goal_tolerance * 4:  # Relaxed tolerance
                    task_completed = True
                    reward += self.success_bonus * 0.8
                    print(f"    ✓ Reached patient area!")
        
        elif 'deliver_medication' in task_name.lower():
            if 'patient_bed' in self.locations and self.medication_retrieved:
                patient_pos = self.locations['patient_bed']
                distance = np.linalg.norm(next_pos - patient_pos)
                
                if distance < self.goal_tolerance * 4:
                    task_completed = True
                    reward += self.success_bonus * 1.5
                    self.medication_delivered = True
                    print(f"    ✓ Medication delivered!")
        
        elif 'move_to_home' in task_name.lower():
            if 'home' in self.locations:
                target_pos = self.locations['home']
                distance = np.linalg.norm(next_pos - target_pos)
                prev_distance = np.linalg.norm(prev_pos - target_pos)
                progress = prev_distance - distance
                
                if distance < self.goal_tolerance * 2:
                    task_completed = True
                    reward += self.success_bonus
                    self.robot_at_home = True
                    print(f"    ✓ Returned home!")
        
        # Progress reward
        if progress > 0:
            reward += progress * 2.0
        
        # Execution quality
        execution_success = execution_data.get('success', False)
        execution_time = execution_data.get('execution_time', 1.0)
        mpc_steps = execution_data.get('steps_executed', 1)
        
        # Time penalty
        reward += self.time_penalty * execution_time
        
        # Efficiency bonus
        if execution_success and mpc_steps < 50:
            efficiency_reward = self.efficiency_bonus * (50 - mpc_steps) / 50.0
            reward += efficiency_reward
        
        # Safety evaluation
        safety_score, safety_violations = self._evaluate_safety(
            next_pos, next_velocity, next_state
        )
        self.episode_safety_violations += safety_violations
        
        # Safety penalty
        safety_penalty = -10.0 * safety_violations
        reward += safety_penalty
        
        # Check delivery completion
        delivery_complete = self._check_delivery_complete()
        if delivery_complete:
            reward += self.success_bonus * 2.0
            print(f"    🎉 FULL DELIVERY COMPLETE!")
        
        # Update episode tracking
        if task_completed:
            self.episode_tasks_completed += 1
        self.episode_rewards.append(reward)
        
        # =====================================================================
        # NEW: PREFERENCE FEATURE ACCUMULATION
        # =====================================================================
        
        self._accumulate_preference_features(
            prev_state=prev_state,
            next_state=next_state,
            target_state=target_state,
            execution_data=execution_data,
            task_completed=task_completed
        )
        
        # =====================================================================
        # CONSTRUCT FEEDBACK
        # =====================================================================
        
        feedback = {
            'reward': float(reward),
            'progress': float(progress),
            'task_completed': task_completed,
            'delivery_complete': delivery_complete,
            'safety_score': safety_score,
            'execution_efficiency': float(1.0 if execution_success else 0.5),
            
            # State tracking
            'medication_retrieved': self.medication_retrieved,
            'medication_delivered': self.medication_delivered,
            'robot_at_home': self.robot_at_home,
            
            # Debug info
            'task_name': task_name,
            'robot_position': next_pos.tolist(),
            'safety_violations': safety_violations,
            
            # NEW: Current feature snapshot
            'current_features': self._get_current_feature_snapshot()
        }
        
        print(f"    Reward: {reward:.3f}, Task complete: {task_completed}")
        
        return feedback
    
    def _accumulate_preference_features(self, 
                                       prev_state: np.ndarray,
                                       next_state: np.ndarray,
                                       target_state: np.ndarray,
                                       execution_data: Dict[str, Any],
                                       task_completed: bool):
        """
        Accumulate features for preference learning.
        
        Maps to 5 preference dimensions:
        - total_time → 'time' preference
        - min_safety_margin → 'safety' preference
        - total_battery_used → 'battery' preference
        - proximity_comfort → 'proximity' preference
        - approach_quality → 'approach' preference
        """
        
        # ===== TIME FEATURE =====
        step_time = execution_data.get('execution_time', 0.2)
        self.episode_features['total_time'] += step_time
        
        # ===== SAFETY FEATURE =====
        # Track minimum safety margin to patient throughout episode
        if 'patient_bed' in self.locations:
            patient_pos = self.locations['patient_bed']
            robot_pos = next_state[:2]
            distance_to_patient = np.linalg.norm(robot_pos - patient_pos)
            
            # Only track when robot is reasonably close to patient
            if distance_to_patient < 5.0:
                current_min = self.episode_features['min_safety_margin']
                self.episode_features['min_safety_margin'] = min(
                    current_min, distance_to_patient
                )
                self.episode_features['patient_proximity_events'] += 1
        
        # ===== BATTERY FEATURE =====
        # Estimate battery usage from control effort
        if 'control_sequence' in execution_data:
            controls = execution_data['control_sequence']
            if controls is not None and len(controls) > 0:
                # Battery ≈ sum of squared accelerations (energy)
                battery_cost = np.sum(controls ** 2) * 0.05
                self.episode_features['total_battery_used'] += battery_cost
        else:
            # Fallback: rough estimate from steps executed
            steps = execution_data.get('steps_executed', 1)
            self.episode_features['total_battery_used'] += steps * 0.02
        
        # ===== PROXIMITY COMFORT FEATURE =====
        # How smoothly did we move near patient?
        if 'patient_bed' in self.locations:
            patient_pos = self.locations['patient_bed']
            robot_pos = next_state[:2]
            distance = np.linalg.norm(robot_pos - patient_pos)
            
            # Only evaluate comfort when near patient
            if distance < 3.0:
                velocity = np.linalg.norm(next_state[3:5])
                
                # Comfort = inverse of velocity when near patient
                # Lower velocity → higher comfort (safer, less scary)
                comfort_score = 1.0 - min(velocity / self.max_safe_velocity, 1.0)
                self.episode_features['proximity_comfort_scores'].append(comfort_score)
        
        # Track velocity profile for analysis
        velocity = np.linalg.norm(next_state[3:5])
        self.episode_features['velocity_profile'].append(velocity)
        
        # ===== APPROACH QUALITY FEATURE =====
        # Evaluate final positioning accuracy when task completes
        if task_completed:
            target_pos = target_state[:2]
            final_pos = next_state[:2]
            position_error = np.linalg.norm(final_pos - target_pos)
            
            # Approach quality = inverse of error
            # Better positioning → higher quality
            max_acceptable_error = 1.5  # meters
            approach_quality = 1.0 - min(position_error / max_acceptable_error, 1.0)
            self.episode_features['approach_quality_scores'].append(approach_quality)
    
    def get_episode_features(self) -> Dict[str, float]:
        """
        Get normalized episode features Φ^(k) for preference learning.
        
        Returns features in [0, 1] range where:
        - 0 = best possible performance
        - 1 = worst acceptable performance
        
        This normalizes across different scales (time in seconds, distance in meters, etc.)
        """
        
        # Extract raw features
        total_time = self.episode_features['total_time']
        min_safety = self.episode_features['min_safety_margin']
        total_battery = self.episode_features['total_battery_used']
        proximity_scores = self.episode_features['proximity_comfort_scores']
        approach_scores = self.episode_features['approach_quality_scores']
        
        # Normalize to [0, 1] where 0 is best
        features = {
            # TIME: Lower is better
            # Normalize to expected range [20s, 120s]
            'time': np.clip(total_time / self.NORM_TIME_MAX, 0.0, 1.0),
            
            # SAFETY: Higher margin is better
            # Normalize margin to [0.5m, 3.0m] range
            # If min_safety is infinity (never near patient), use 0 (best safety)
            'safety': 0.0 if min_safety == float('inf') else (
                1.0 - np.clip(
                    (min_safety - self.NORM_SAFETY_MIN) / 
                    (self.NORM_SAFETY_MAX - self.NORM_SAFETY_MIN),
                    0.0, 1.0
                )
            ),
            
            # BATTERY: Lower usage is better
            # Normalize to expected range [0, 100 units]
            'battery': np.clip(total_battery / self.NORM_BATTERY_MAX, 0.0, 1.0),
            
            # PROXIMITY: Higher comfort is better
            # Average of comfort scores (already in [0,1], 1=comfortable)
            # Convert so 0=best (comfortable), 1=worst (uncomfortable)
            'proximity': (
                1.0 - np.mean(proximity_scores) 
                if proximity_scores else 0.5  # Neutral if no data
            ),
            
            # APPROACH: Higher quality is better
            # Average of approach quality scores (already in [0,1], 1=good)
            # Convert so 0=best (accurate), 1=worst (inaccurate)
            'approach': (
                1.0 - np.mean(approach_scores) 
                if approach_scores else 0.5  # Neutral if no data
            )
        }
        
        print(f"\n   Episode Features (normalized [0=best, 1=worst]):")
        print(f"     Time: {features['time']:.3f} ({total_time:.1f}s)")
        print(f"     Safety: {features['safety']:.3f} (margin: {min_safety:.2f}m)")
        print(f"     Battery: {features['battery']:.3f} ({total_battery:.1f} units)")
        print(f"     Proximity: {features['proximity']:.3f} (n={len(proximity_scores)})")
        print(f"     Approach: {features['approach']:.3f} (n={len(approach_scores)})")
        
        return features
    
    def _get_current_feature_snapshot(self) -> Dict[str, Any]:
        """Get current raw feature values (for debugging)."""
        return {
            'total_time': self.episode_features['total_time'],
            'min_safety_margin': self.episode_features['min_safety_margin'],
            'total_battery': self.episode_features['total_battery_used'],
            'proximity_events': len(self.episode_features['proximity_comfort_scores']),
            'approach_events': len(self.episode_features['approach_quality_scores'])
        }
    
    def _evaluate_safety(self, position: np.ndarray, velocity: float, 
                        full_state: np.ndarray) -> tuple:
        """Evaluate safety metrics (existing logic)."""
        
        safety_score = 1.0
        violations = 0
        
        # Check velocity limits
        if velocity > self.max_safe_velocity:
            safety_score -= 0.2
            violations += 1
        
        # Check patient safety zone
        if 'patient_bed' in self.locations:
            patient_pos = self.locations['patient_bed']
            distance_to_patient = np.linalg.norm(position - patient_pos)
            
            if distance_to_patient < self.patient_safety_radius:
                if velocity > self.min_safe_velocity * 2:
                    safety_score -= 0.3
                    violations += 1
        
        # Check for erratic orientation changes
        angular_velocity = abs(full_state[5]) if len(full_state) > 5 else 0.0
        if angular_velocity > 2.0:
            safety_score -= 0.1
            violations += 1
        
        return max(safety_score, 0.0), violations
    
    def _check_delivery_complete(self) -> bool:
        """Check if complete delivery mission is accomplished."""
        return (self.medication_retrieved and 
                self.medication_delivered and 
                self.robot_at_home)
    
    def finalize_episode(self) -> Dict[str, Any]:
        """Finalize episode and return summary with features."""
        
        # Get normalized features for preference learning
        normalized_features = self.get_episode_features()
        
        episode_summary = {
            'total_reward': sum(self.episode_rewards),
            'tasks_completed': self.episode_tasks_completed,
            'safety_violations': self.episode_safety_violations,
            'delivery_success': self._check_delivery_complete(),
            'medication_retrieved': self.medication_retrieved,
            'medication_delivered': self.medication_delivered,
            'robot_at_home': self.robot_at_home,
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            
            # NEW: Include preference features
            'preference_features': normalized_features,
            'raw_features': self.episode_features.copy()
        }
        
        print(f"\n[INFO] Episode Summary:")
        print(f"   Tasks completed: {self.episode_tasks_completed}")
        print(f"   Delivery success: {episode_summary['delivery_success']}")
        print(f"   Total reward: {episode_summary['total_reward']:.2f}")
        
        return episode_summary
    
    def reset_episode(self):
        """Reset episode state tracking including preference features."""
        
        # Reset task tracking
        self.medication_retrieved = False
        self.medication_delivered = False
        self.robot_at_home = False
        self.episode_rewards = []
        self.episode_safety_violations = 0
        self.episode_tasks_completed = 0
        
        # Reset preference features
        self.episode_features = {
            'total_time': 0.0,
            'min_safety_margin': float('inf'),
            'total_battery_used': 0.0,
            'proximity_comfort_scores': [],
            'approach_quality_scores': [],
            'velocity_profile': [],
            'patient_proximity_events': 0
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        
        return {
            'current_episode_reward': sum(self.episode_rewards),
            'tasks_completed': self.episode_tasks_completed,
            'safety_violations': self.episode_safety_violations,
            'medication_status': {
                'retrieved': self.medication_retrieved,
                'delivered': self.medication_delivered,
                'robot_home': self.robot_at_home
            },
            'delivery_progress': self._get_delivery_progress(),
            'current_features': self._get_current_feature_snapshot()
        }
    
    def _get_delivery_progress(self) -> float:
        """Calculate delivery progress as percentage."""
        
        progress = 0.0
        if self.medication_retrieved:
            progress += 0.4
        if self.medication_delivered:
            progress += 0.4
        if self.robot_at_home:
            progress += 0.2
            
        return progress