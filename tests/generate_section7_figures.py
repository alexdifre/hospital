#!/usr/bin/env python3
"""
Generate Section 7 (Experimental Setup) figures for the MLC Stack paper.

Figures:
  A1. Hospital Environment Floor Plan
  A2. Medication Delivery State Diagram
  A3. Meal Preparation State Diagram
  A4. Patient Profile Weight Vectors
  A5. Fuzzy Membership Functions
  A6. System Architecture Diagram

Usage:
  python generate_section7_figures.py          # all figures
  python generate_section7_figures.py A1 A4    # specific figures
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# =====================================================================
# STYLING
# =====================================================================

STYLE = {
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
}
plt.rcParams.update(STYLE)

PROFILE_COLORS = {
    "speed_oriented":       "#2196F3",
    "safety_first":         "#F44336",
    "presentation_focused": "#4CAF50",
    "comfort_focused":      "#FF9800",
    "energy_conscious":     "#9C27B0",
}

DIM_COLORS = {
    "time":      "#2196F3",
    "safety":    "#F44336",
    "battery":   "#FF9800",
    "proximity": "#4CAF50",
    "approach":  "#9C27B0",
}

DIM_NAMES = ["time", "safety", "battery", "proximity", "approach"]

# =====================================================================
# DATA
# =====================================================================

LOCATIONS = {
    "home":              (0, 0),
    "pharmacy_north":    (5, 18),
    "pharmacy_south":    (-5, -10),
    "supply_A":          (14, 10),
    "supply_B":          (15, -12),
    "patient_bed":       (20, 12),
    "patient_bed_left":  (20.5, 12),
    "patient_bed_right": (19.5, 12),
    "nurse_station":     (3, 5),
    "equipment_storage": (10, -5),
    "charge_main":       (8, 8),
    "charge_backup":     (-2, 5),
    "pantry":            (-3, 15),
    "prep_station":      (0, 20),
    "stove":             (2, 22),
}

RISK_MAP = {
    "nurse_station": 0.60, "stove": 0.70, "equipment_storage": 0.40,
    "pharmacy_north": 0.30, "supply_B": 0.30, "prep_station": 0.30,
    "patient_bed_left": 0.15, "patient_bed_right": 0.15, "pantry": 0.15,
    "charge_backup": 0.08, "pharmacy_south": 0.05, "supply_A": 0.05,
    "charge_main": 0.05, "home": 0.02, "patient_bed": 0.15,
}

LOC_CATEGORIES = {
    "pharmacy":  ["pharmacy_north", "pharmacy_south"],
    "supply":    ["supply_A", "supply_B"],
    "kitchen":   ["pantry", "prep_station", "stove"],
    "patient":   ["patient_bed", "patient_bed_left", "patient_bed_right"],
    "charge":    ["charge_main", "charge_backup"],
    "other":     ["home", "nurse_station", "equipment_storage"],
}

CAT_COLORS = {
    "pharmacy": "#E53935", "supply": "#1E88E5", "kitchen": "#FB8C00",
    "patient":  "#43A047", "charge": "#FDD835", "other":   "#757575",
}

CAT_MARKERS = {
    "pharmacy": "s", "supply": "D", "kitchen": "^",
    "patient":  "*", "charge": "p", "other":   "o",
}

PATIENT_PROFILES = {
    "speed_oriented":       [0.50, 0.12, 0.14, 0.14, 0.10],
    "safety_first":         [0.10, 0.50, 0.15, 0.15, 0.10],
    "presentation_focused": [0.05, 0.10, 0.05, 0.20, 0.60],
    "comfort_focused":      [0.15, 0.15, 0.10, 0.40, 0.20],
    "energy_conscious":     [0.15, 0.15, 0.45, 0.15, 0.10],
}

PROFILE_LABELS = {
    "speed_oriented":       "Speed-Oriented",
    "safety_first":         "Safety-First",
    "presentation_focused": "Presentation-Focused",
    "comfort_focused":      "Comfort-Focused",
    "energy_conscious":     "Energy-Conscious",
}

OUTPUT_DIR = Path("/home/claude/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# A1: HOSPITAL ENVIRONMENT FLOOR PLAN
# =====================================================================

def fig_a1_floor_plan():
    fig, ax = plt.subplots(figsize=(9, 10))

    # Risk zones (background circles with opacity)
    for name, (x, y) in LOCATIONS.items():
        risk = RISK_MAP.get(name, 0.1)
        if risk > 0.1:
            radius = 1.5 + risk * 3.0
            circle = plt.Circle(
                (x, y), radius, color="red",
                alpha=risk * 0.25, zorder=1,
            )
            ax.add_patch(circle)

    # Kitchen cluster boundary
    kitchen_rect = FancyBboxPatch(
        (-5.5, 13.5), 10, 10.5,
        boxstyle="round,pad=0.5",
        facecolor="#FFF3E0", edgecolor="#FB8C00",
        linewidth=1.5, linestyle="--", zorder=0, alpha=0.5,
    )
    ax.add_patch(kitchen_rect)
    ax.text(-4.5, 23.5, "Kitchen Area", fontsize=9, color="#E65100",
            fontstyle="italic", fontweight="bold")

    # Patient zone boundary
    patient_rect = FancyBboxPatch(
        (17.5, 10), 5, 4,
        boxstyle="round,pad=0.5",
        facecolor="#E8F5E9", edgecolor="#43A047",
        linewidth=1.5, linestyle="--", zorder=0, alpha=0.5,
    )
    ax.add_patch(patient_rect)
    ax.text(17.8, 13.7, "Patient Zone", fontsize=9, color="#2E7D32",
            fontstyle="italic", fontweight="bold")

    # Plot locations by category
    for cat, names in LOC_CATEGORIES.items():
        color = CAT_COLORS[cat]
        marker = CAT_MARKERS[cat]
        for name in names:
            if name not in LOCATIONS:
                continue
            x, y = LOCATIONS[name]
            size = 120 if cat == "patient" else 80
            ax.scatter(x, y, c=color, marker=marker, s=size,
                      edgecolors="black", linewidths=0.8, zorder=5)

            # Label positioning (avoid overlaps)
            offsets = {
                "home": (-1.5, -1.5), "pharmacy_north": (-6, 0.5),
                "pharmacy_south": (-6, -1.5), "supply_A": (1.2, -1.2),
                "supply_B": (1.2, -1.2), "patient_bed": (0, 1.5),
                "patient_bed_left": (-2, -1.8), "patient_bed_right": (-2.5, 1.5),
                "nurse_station": (-5.5, -1.2), "equipment_storage": (-5, -1.5),
                "charge_main": (1.2, -1.2), "charge_backup": (-5, -1.2),
                "pantry": (-4.5, -1.2), "prep_station": (1.2, 0),
                "stove": (1.2, 0.5),
            }
            dx, dy = offsets.get(name, (1.2, 0))
            display_name = name.replace("_", " ").title()
            # Shorten some names
            short = {
                "Patient Bed Left": "Bed (L)",
                "Patient Bed Right": "Bed (R)",
                "Patient Bed": "Patient Bed",
                "Pharmacy North": "Pharmacy N",
                "Pharmacy South": "Pharmacy S",
                "Equipment Storage": "Equip. Storage",
                "Charge Main": "Charge (Main)",
                "Charge Backup": "Charge (Backup)",
            }
            display_name = short.get(display_name, display_name)
            fontsize = 8 if "Bed" in display_name else 9
            ax.annotate(
                display_name, (x, y), (x + dx, y + dy),
                fontsize=fontsize, ha="left" if dx > 0 else "right",
                va="center",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)
                if abs(dx) > 2 or abs(dy) > 2 else None,
            )

            # Risk label
            risk = RISK_MAP.get(name, 0)
            if risk >= 0.30:
                ax.text(x, y - 1.0, f"risk={risk:.2f}",
                       fontsize=7, ha="center", color="#B71C1C",
                       fontstyle="italic")

    # Legend
    legend_handles = []
    for cat in ["pharmacy", "supply", "kitchen", "patient", "charge", "other"]:
        h = plt.scatter([], [], c=CAT_COLORS[cat], marker=CAT_MARKERS[cat],
                       s=60, edgecolors="black", linewidths=0.5,
                       label=cat.title())
        legend_handles.append(h)
    # Risk zone legend entry
    risk_patch = mpatches.Patch(color="red", alpha=0.15, label="Risk zone (r ≥ 0.15)")
    legend_handles.append(risk_patch)
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
             framealpha=0.9, edgecolor="gray")

    # Grid and labels
    ax.set_xlim(-10, 27)
    ax.set_ylim(-17, 26)
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")
    ax.set_title("Hospital Environment — Location Map & Risk Zones")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Scale bar
    ax.plot([20, 25], [-15, -15], "k-", linewidth=2)
    ax.text(22.5, -16, "5 m", ha="center", fontsize=9)

    save(fig, "fig7_floor_plan")


# =====================================================================
# A2: MEDICATION DELIVERY STATE DIAGRAM
# =====================================================================

def fig_a2_med_state_diagram():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 7.5)
    ax.axis("off")
    ax.set_title("Medication Delivery — Task State Transitions", fontsize=14, pad=15)

    # State boxes
    def state_box(x, y, text, color="#E3F2FD", width=1.8, height=0.7):
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.12", facecolor=color,
            edgecolor="#1565C0", linewidth=1.5, zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=9,
               fontweight="bold", zorder=4)

    # Action labels on arrows
    def action_arrow(x1, y1, x2, y2, text, color="#1565C0", curve=0):
        style = f"arc3,rad={curve}" if curve else "arc3,rad=0"
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=1.5,
                connectionstyle=style,
            ),
            zorder=2,
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if curve:
            my += curve * 0.6
        ax.text(mx, my + 0.25, text, ha="center", va="bottom",
               fontsize=8, color=color, fontstyle="italic")

    # === Layout ===
    # Row 0 (top): START
    state_box(1, 6.5, "START\n(home)", "#E8F5E9")

    # Row 1: Supply choice (branching)
    state_box(3, 6.5, "Go to\nSupply", "#FFF3E0")
    action_arrow(1.9, 6.5, 2.1, 6.5, "")

    state_box(1.5, 4.8, "Supply A", "#BBDEFB")
    state_box(4.5, 4.8, "Supply B", "#BBDEFB")
    action_arrow(3, 5.8, 1.8, 5.2, "go_to_supply_a", color="#1565C0", curve=-0.1)
    action_arrow(3, 5.8, 4.2, 5.2, "go_to_supply_b", color="#1565C0", curve=0.1)

    # Collect supplement
    state_box(1.5, 3.5, "Has\nSupplement", "#C8E6C9")
    state_box(4.5, 3.5, "Has\nSupplement", "#C8E6C9")
    action_arrow(1.5, 4.4, 1.5, 4.0, "collect_supplement")
    action_arrow(4.5, 4.4, 4.5, 4.0, "collect_supplement")

    # Pharmacy choice (branching)
    state_box(6.5, 4.8, "Go to\nPharmacy", "#FFF3E0")

    # Merge arrows to pharmacy choice
    action_arrow(2.4, 3.5, 5.6, 4.8, "", color="#78909C")
    action_arrow(5.4, 3.5, 5.6, 4.8, "", color="#78909C")

    state_box(5.5, 3.2, "Pharmacy N\n(risk 0.30)", "#FFCDD2")
    state_box(7.5, 3.2, "Pharmacy S\n(risk 0.05)", "#C8E6C9")
    action_arrow(6.2, 4.1, 5.8, 3.7, "go_to_pharm_n", color="#E53935", curve=-0.1)
    action_arrow(6.8, 4.1, 7.2, 3.7, "go_to_pharm_s", color="#43A047", curve=0.1)

    # Collect medication
    state_box(5.5, 1.8, "Has Med\n+ Suppl", "#C8E6C9")
    state_box(7.5, 1.8, "Has Med\n+ Suppl", "#C8E6C9")
    action_arrow(5.5, 2.8, 5.5, 2.3, "collect_med")
    action_arrow(7.5, 2.8, 7.5, 2.3, "collect_med")

    # Optional recharge
    state_box(3, 1.8, "Recharge\n(optional)", "#FFF9C4")
    ax.annotate(
        "", xy=(3.9, 1.8), xytext=(4.6, 1.8),
        arrowprops=dict(arrowstyle="<|-", color="#FBC02D", lw=1.5,
                       connectionstyle="arc3,rad=0", linestyle="--"),
        zorder=2,
    )
    ax.text(3.9, 2.15, "recharge", ha="center", fontsize=7,
           color="#F57F17", fontstyle="italic")

    # Approach choice (branching)
    state_box(6.5, 0.5, "Go to\nPatient", "#FFF3E0")
    action_arrow(5.8, 1.4, 6.2, 1.0, "", color="#78909C")
    action_arrow(7.2, 1.4, 6.8, 1.0, "", color="#78909C")

    state_box(8.5, 1.8, "Left\nApproach", "#E1F5FE")
    state_box(8.5, 0.5, "Right\nApproach", "#E1F5FE")
    action_arrow(7.4, 0.7, 7.6, 1.5, "go_patient_L", color="#0277BD", curve=-0.3)
    action_arrow(7.4, 0.3, 7.6, 0.5, "go_patient_R", color="#0277BD", curve=0.1)

    # Deliver
    state_box(10, 1.1, "DELIVERED", "#A5D6A7", width=1.6)
    action_arrow(9.2, 1.8, 9.5, 1.4, "deliver", color="#2E7D32")
    action_arrow(9.2, 0.5, 9.5, 0.8, "deliver", color="#2E7D32")

    # Annotation box: plan skeleton
    ax.text(0.5, 0.3,
           "Plan skeleton: 2 supply × 2 pharmacy × 2 approach = 8 routes\n"
           "+ optional recharge insertion → 24 combinations",
           fontsize=9, fontstyle="italic", color="#546E7A",
           bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECEFF1",
                    edgecolor="#90A4AE", alpha=0.9))

    save(fig, "fig7_med_state_diagram")


# =====================================================================
# A3: MEAL PREPARATION STATE DIAGRAM
# =====================================================================

def fig_a3_meal_state_diagram():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-0.5, 8.5)
    ax.axis("off")
    ax.set_title("Meal Preparation — Task State Transitions", fontsize=14, pad=15)

    def state_box(x, y, text, color="#E3F2FD", width=1.7, height=0.65):
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.10", facecolor=color,
            edgecolor="#1565C0", linewidth=1.5, zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5,
               fontweight="bold", zorder=4)

    def action_arrow(x1, y1, x2, y2, text, color="#1565C0", curve=0, fontsize=7.5):
        style = f"arc3,rad={curve}" if curve else "arc3,rad=0"
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                           connectionstyle=style), zorder=2,
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if curve:
            my += curve * 0.5
        if text:
            ax.text(mx, my + 0.22, text, ha="center", va="bottom",
                   fontsize=fontsize, color=color, fontstyle="italic")

    # === START ===
    state_box(1, 7.5, "START\n(pantry)", "#E8F5E9")

    # === MEAL TYPE BRANCHING ===
    # Sandwich path (top)
    state_box(3.5, 7.5, "Collect\nSandwich Ingr.", "#FFECB3")
    action_arrow(1.85, 7.5, 2.65, 7.5, "")

    # Soup path (middle)
    state_box(3.5, 5.8, "Collect\nSoup Ingr.", "#FFE0B2")
    action_arrow(1.6, 7.1, 2.65, 6.1, "", color="#FB8C00", curve=-0.1)

    # Full meal path (bottom)
    state_box(3.5, 4.1, "Collect\nFull Meal Ingr.", "#FFCCBC")
    action_arrow(1.4, 7.0, 2.65, 4.4, "", color="#E64A19", curve=-0.15)

    # Labels for branching
    ax.text(2.1, 7.9, "sandwich", fontsize=8, color="#F57F17", fontstyle="italic")
    ax.text(1.8, 6.4, "soup", fontsize=8, color="#FB8C00", fontstyle="italic")
    ax.text(1.5, 5.3, "full meal", fontsize=8, color="#E64A19", fontstyle="italic")

    # === SANDWICH PATH (top) ===
    state_box(5.5, 7.5, "Go to\nPrep Station", "#E3F2FD")
    action_arrow(4.35, 7.5, 4.65, 7.5, "")

    state_box(7.5, 7.5, "Assemble", "#C8E6C9")
    action_arrow(6.35, 7.5, 6.65, 7.5, "")

    # === SOUP PATH (middle) ===
    state_box(5.5, 5.8, "Go to\nPrep Station", "#E3F2FD")
    action_arrow(4.35, 5.8, 4.65, 5.8, "")

    state_box(7.5, 5.8, "Chop", "#DCEDC8")
    action_arrow(6.35, 5.8, 6.65, 5.8, "")

    state_box(9, 5.8, "Go to\nStove", "#E3F2FD")
    action_arrow(8.35, 5.8, 8.15, 5.8, "")

    state_box(10.5, 5.8, "Cook", "#DCEDC8")
    action_arrow(9.85, 5.8, 9.65, 5.8, "")

    # === FULL MEAL PATH (bottom) ===
    state_box(5.5, 4.1, "Go to\nPrep Station", "#E3F2FD")
    action_arrow(4.35, 4.1, 4.65, 4.1, "")

    state_box(7.5, 4.1, "Chop", "#DCEDC8")
    action_arrow(6.35, 4.1, 6.65, 4.1, "")

    state_box(9, 4.1, "Go to\nStove", "#E3F2FD")
    action_arrow(8.35, 4.1, 8.15, 4.1, "")

    state_box(10.5, 4.1, "Cook", "#DCEDC8")
    action_arrow(9.85, 4.1, 9.65, 4.1, "")

    # Full meal extra: return to prep + plate
    state_box(10.5, 2.6, "Return to\nPrep Station", "#E3F2FD")
    action_arrow(10.5, 3.7, 10.5, 3.0, "")

    state_box(10.5, 1.3, "Plate", "#DCEDC8")
    action_arrow(10.5, 2.3, 10.5, 1.7, "")

    # === CONVERGENCE: Go to Patient ===
    state_box(10.5, 7.5, "READY", "#C8E6C9", width=1.3, height=0.55)
    # Sandwich → ready
    action_arrow(8.35, 7.5, 9.55, 7.5, "", color="#2E7D32")
    # Soup → ready (after cook, go back to prep for implicit ready)
    action_arrow(11.0, 5.5, 11.3, 7.2, "", color="#2E7D32", curve=0.3)
    # Full meal → ready (after plate)
    action_arrow(11.0, 1.3, 11.8, 7.2, "", color="#2E7D32", curve=0.4)

    # Approach choice
    state_box(12, 6.2, "Go Patient\n(Left)", "#E1F5FE", width=1.5)
    state_box(12, 5.0, "Go Patient\n(Right)", "#E1F5FE", width=1.5)
    action_arrow(11.1, 7.2, 11.5, 6.5, "L", color="#0277BD", curve=-0.1)
    action_arrow(11.1, 7.1, 11.5, 5.3, "R", color="#0277BD", curve=-0.2)

    # Deliver
    state_box(12, 3.5, "DELIVERED", "#A5D6A7", width=1.5)
    action_arrow(12, 5.7, 12, 5.35, "", color="#2E7D32")
    action_arrow(12, 4.7, 12, 3.9, "deliver_meal", color="#2E7D32")

    # Complexity annotations
    sandbox_box = dict(boxstyle="round,pad=0.3", facecolor="#ECEFF1",
                      edgecolor="#90A4AE", alpha=0.9)
    ax.text(0.5, 2.5,
           "Sandwich: 5 steps (fastest, lowest approach bonus)\n"
           "Soup: 8 steps (chop+cook, moderate approach bonus)\n"
           "Full Meal: 9 steps (chop+cook+plate, highest approach bonus +0.20)",
           fontsize=8.5, fontstyle="italic", color="#546E7A",
           bbox=sandbox_box, verticalalignment="top")

    # Time annotations on paths
    ax.text(7.5, 8.0, "~34s, 5 steps", fontsize=7, ha="center",
           color="#757575")
    ax.text(9, 6.3, "~55s, 8 steps", fontsize=7, ha="center",
           color="#757575")
    ax.text(7.5, 3.5, "~82s, 9 steps", fontsize=7, ha="center",
           color="#757575")

    save(fig, "fig7_meal_state_diagram")


# =====================================================================
# A4: PATIENT PROFILE WEIGHT VECTORS
# =====================================================================

def fig_a4_profile_weights():
    fig, ax = plt.subplots(figsize=(9, 5))

    profiles = list(PATIENT_PROFILES.keys())
    n_profiles = len(profiles)
    n_dims = 5
    x = np.arange(n_dims)
    bar_width = 0.14
    offsets = np.arange(n_profiles) - (n_profiles - 1) / 2

    for i, profile in enumerate(profiles):
        weights = PATIENT_PROFILES[profile]
        color = PROFILE_COLORS[profile]
        label = PROFILE_LABELS[profile]
        bars = ax.bar(
            x + offsets[i] * bar_width, weights, bar_width,
            color=color, edgecolor="white", linewidth=0.5,
            label=label, alpha=0.85,
        )
        # Value labels on dominant weight
        for j, (bar, w) in enumerate(zip(bars, weights)):
            if w == max(weights):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f"{w:.0%}", ha="center", va="bottom", fontsize=7,
                       fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in DIM_NAMES], fontsize=11)
    ax.set_ylabel("Weight (w*)")
    ax.set_title("Patient Profile Preference Weights (Ground Truth)")
    ax.set_ylim(0, 0.72)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "fig7_profile_weights")


# =====================================================================
# A5: FUZZY MEMBERSHIP FUNCTIONS
# =====================================================================

def fig_a5_fuzzy_membership():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- (a) Location membership Gaussians ---
    ax = axes[0]
    distances = np.linspace(0, 8, 200)

    # Gaussian membership: μ(d) = exp(-d² / (2σ²))
    # σ varies by location type
    location_sigmas = {
        "Pharmacy N (σ=2.5)": 2.5,
        "Supply A (σ=2.0)":   2.0,
        "Patient Bed (σ=1.5)": 1.5,
        "Home (σ=3.0)":       3.0,
    }

    colors_loc = ["#E53935", "#1E88E5", "#43A047", "#757575"]
    for (name, sigma), color in zip(location_sigmas.items(), colors_loc):
        membership = np.exp(-distances**2 / (2 * sigma**2))
        ax.plot(distances, membership, color=color, linewidth=2, label=name)

    # Threshold line
    ax.axhline(y=0.3, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(7.5, 0.32, "threshold", fontsize=8, color="gray", ha="right")

    ax.set_xlabel("Distance from location center (m)")
    ax.set_ylabel("Membership μ(d)")
    ax.set_title("(a) Location Membership Functions")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- (b) Battery membership triangles ---
    ax = axes[1]
    soc = np.linspace(0, 1, 200)

    # Triangular membership functions
    # Low: peaks at 0, reaches 0 at 0.4
    mu_low = np.clip(1.0 - soc / 0.4, 0, 1)
    # Medium: peaks at 0.5, zero below 0.2, zero above 0.8
    mu_med = np.clip(np.minimum((soc - 0.2) / 0.3, (0.8 - soc) / 0.3), 0, 1)
    # High: peaks at 1, reaches 0 at 0.6
    mu_high = np.clip((soc - 0.6) / 0.4, 0, 1)

    ax.fill_between(soc, mu_low, alpha=0.15, color="#F44336")
    ax.fill_between(soc, mu_med, alpha=0.15, color="#FF9800")
    ax.fill_between(soc, mu_high, alpha=0.15, color="#4CAF50")
    ax.plot(soc, mu_low, color="#F44336", linewidth=2, label="Low")
    ax.plot(soc, mu_med, color="#FF9800", linewidth=2, label="Medium")
    ax.plot(soc, mu_high, color="#4CAF50", linewidth=2, label="High")

    # Example annotation: SoC=0.35
    soc_ex = 0.35
    ax.axvline(x=soc_ex, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(soc_ex + 0.02, 0.95, f"SoC = {soc_ex}", fontsize=8,
           color="#424242", fontstyle="italic")
    # Show membership values at example point
    ml = float(np.clip(1.0 - soc_ex / 0.4, 0, 1))
    mm = float(np.clip(min((soc_ex - 0.2) / 0.3, (0.8 - soc_ex) / 0.3), 0, 1))
    ax.plot(soc_ex, ml, "o", color="#F44336", markersize=6, zorder=5)
    ax.plot(soc_ex, mm, "o", color="#FF9800", markersize=6, zorder=5)

    ax.set_xlabel("State of Charge (SoC)")
    ax.set_ylabel("Membership μ(SoC)")
    ax.set_title("(b) Battery Level Membership Functions")
    ax.legend(fontsize=9, loc="center right")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save(fig, "fig7_fuzzy_membership")


# =====================================================================
# A6: SYSTEM ARCHITECTURE DIAGRAM
# =====================================================================

def fig_a6_architecture():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, 8)
    ax.axis("off")
    ax.set_title("Dual-Loop Learning Architecture", fontsize=14, pad=15)

    def block(x, y, w, h, text, color="#E3F2FD", edge="#1565C0",
              fontsize=9, bold=True):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15", facecolor=color,
            edgecolor=edge, linewidth=1.8, zorder=3,
        )
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
               fontsize=fontsize, fontweight=weight, zorder=4,
               linespacing=1.4)

    def arrow(x1, y1, x2, y2, text="", color="#424242", curve=0,
              style="-|>", lw=1.5, fontsize=8):
        cs = f"arc3,rad={curve}" if curve else "arc3,rad=0"
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                           connectionstyle=cs), zorder=2,
        )
        if text:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            offset = 0.18
            if curve:
                my += curve * 0.8
            ax.text(mx, my + offset, text, ha="center", va="bottom",
                   fontsize=fontsize, color=color, fontstyle="italic")

    # === OUTER LOOP (top) ===
    # Loop label
    outer_rect = FancyBboxPatch(
        (-0.3, 3.8), 12.2, 3.9,
        boxstyle="round,pad=0.2", facecolor="#E8F5E9",
        edgecolor="#43A047", linewidth=2, linestyle="--",
        zorder=0, alpha=0.3,
    )
    ax.add_patch(outer_rect)
    ax.text(0, 7.5, "OUTER LOOP — Preference Learning",
           fontsize=11, fontweight="bold", color="#2E7D32")

    # Task Planner
    block(0, 5.5, 2.2, 1.1,
          "Task Planner\n(PDDL/ENHSP-opt)", "#C8E6C9", "#2E7D32")

    # MPC / Execution
    block(3.2, 5.5, 2.2, 1.1,
          "Execute Plan\n(MPC + Nav)", "#BBDEFB", "#1565C0")
    arrow(2.2, 6.05, 3.2, 6.05, "actions", "#2E7D32")

    # Feature Extraction
    block(6.4, 5.5, 2.2, 1.1,
          "Feature\nExtraction", "#FFF3E0", "#FB8C00")
    arrow(5.4, 6.05, 6.4, 6.05, "trajectory", "#1565C0")

    # Patient Rating
    block(6.4, 4.0, 2.2, 1.0,
          "Patient\nRating r(f)", "#FFECB3", "#F57F17")
    arrow(7.5, 5.5, 7.5, 5.0, "features f", "#FB8C00")

    # Weight Update
    block(9.5, 4.0, 2.0, 1.0,
          "w Update\n(simplex proj.)", "#C8E6C9", "#2E7D32")
    arrow(8.6, 4.5, 9.5, 4.5, "ratings", "#F57F17")

    # Feedback arrow: w → planner
    arrow(10.5, 5.0, 10.5, 6.05, "", "#2E7D32", curve=0)
    arrow(10.5, 6.5, 2.2, 6.5, "updated w", "#2E7D32", curve=0.25)

    # === TRANSLATOR CONFIG (bottom) ===
    inner_rect = FancyBboxPatch(
        (1.7, -0.2), 7.8, 3.5,
        boxstyle="round,pad=0.2", facecolor="#E3F2FD",
        edgecolor="#1565C0", linewidth=2, linestyle="--",
        zorder=0, alpha=0.3,
    )
    ax.add_patch(inner_rect)
    ax.text(2.0, 3.1, "TERMINAL TARGET LEARNING",
           fontsize=11, fontweight="bold", color="#1565C0")

    # Translator
    block(2, 1.5, 2.2, 1.1,
          "Translator φ\n(fixed Q, R, N)", "#BBDEFB", "#1565C0")

    # MPC (shared with outer)
    block(5, 1.5, 2.0, 1.1,
          "HybridMPC\n(Acados)", "#E1F5FE", "#0277BD")
    arrow(4.2, 2.05, 5.0, 2.05, "Q, R", "#1565C0")

    # Sensitivities
    block(5, 0.0, 2.0, 1.0,
          "IFT\nstandalone only", "#FCE4EC", "#C62828")
    arrow(6.0, 1.5, 6.0, 1.0, "offline", "#0277BD")

    # z_target update
    block(2, 0.0, 2.2, 1.0,
          "z_target Update\n(JᵀE)", "#BBDEFB", "#1565C0")
    arrow(5.0, 0.5, 4.2, 0.5, "J_E,z", "#C62828")

    # Disabled update path
    arrow(3.1, 1.0, 3.1, 1.5, "target", "#1565C0")

    # Connection: planner/execution uses fixed translator parameters
    arrow(4.3, 5.5, 4.3, 2.6, "fixed MPC\nparams",
          "#78909C", curve=0.2)
    arrow(3.1, 2.6, 3.1, 5.5, "control\nparams",
          "#78909C", curve=0.2)

    # === ENVIRONMENT ===
    block(9.5, 1.5, 2.0, 1.1,
          "MuJoCo\nEnvironment", "#F3E5F5", "#7B1FA2")
    arrow(7.0, 2.05, 9.5, 2.05, "controls u", "#0277BD")
    arrow(9.5, 2.6, 7.0, 5.5, "state x", "#7B1FA2", curve=-0.3)

    # === FUZZY BRIDGE ===
    block(9.5, 5.5, 2.0, 1.1,
          "Fuzzy State\nEstimator", "#F3E5F5", "#7B1FA2")
    arrow(9.5, 2.9, 10.2, 5.5, "position", "#7B1FA2", curve=0.15)
    arrow(9.5, 5.8, 2.2, 5.8, "memberships", "#7B1FA2", curve=0.15)

    save(fig, "fig7_architecture")


# =====================================================================
# UTILITY
# =====================================================================

def save(fig, name):
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    png_path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(str(pdf_path), format="pdf")
    fig.savefig(str(png_path), format="png")
    plt.close(fig)
    print(f"  ✓ {name} → {pdf_path}")


# =====================================================================
# MAIN
# =====================================================================

FIGURE_MAP = {
    "A1": ("Hospital Floor Plan",           fig_a1_floor_plan),
    "A2": ("Medication State Diagram",      fig_a2_med_state_diagram),
    "A3": ("Meal Preparation State Diagram", fig_a3_meal_state_diagram),
    "A4": ("Patient Profile Weights",       fig_a4_profile_weights),
    "A5": ("Fuzzy Membership Functions",    fig_a5_fuzzy_membership),
    "A6": ("System Architecture Diagram",   fig_a6_architecture),
}

if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(FIGURE_MAP.keys())

    print(f"Generating {len(targets)} Section 7 figures...\n")
    for key in targets:
        if key in FIGURE_MAP:
            desc, func = FIGURE_MAP[key]
            print(f"[{key}] {desc}")
            func()
        else:
            print(f"[{key}] Unknown figure — skipping")

    print(f"\nDone. Figures saved to {OUTPUT_DIR}/")
