"""



NON CONSIDERARE QUESTO FILE!!!!



USA PIUTTOSTO script_benchmark/extract_joint_cmd_vs_state.py



"""












#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract selected topics/fields from a ROS 2 MCAP bag and generate CSVs + plots.

Requirements (Python >=3.9):
    pip install mcap mcap-ros2-support pandas matplotlib

Usage:
    python extract_mcap_timeseries.py --mcap path/to/bag.mcap --out out_dir

What it does:
- Reads ROS 2 messages from the MCAP.
- Extracts the following signals (with field paths):
    /cmd_vel/angular/x
    /cmd_vel/angular/y
    /cmd_vel/angular/z
    /cmd_vel/linear/x
    /cmd_vel/linear/y
    /cmd_vel/linear/z
    /rigid_bodies/rigidbodies[3]/pose/position/x
    /rigid_bodies/rigidbodies[3]/pose/position/y
    /rigid_bodies/rigidbodies[3]/pose/position/z
    /state_broadcaster/joints_state/velocity[10]
    /state_broadcaster/joints_state/velocity[11]
    /state_broadcaster/joints_state/velocity[8]
    /state_broadcaster/joints_state/velocity[9]
- Saves three CSVs:
    cmd_vel.csv, rigid_body_pos.csv, joint_velocities.csv
- Creates three matplotlib figures: cmd_vel.png, rigid_body_pos.png, joint_velocities.png

Notes:
- Timestamps use the MCAP log time (ns). We convert to seconds and shift so t=0 at first message.
- Field-paths support nested attributes with "/" and array indices with "[i]".
"""

import argparse
import os
import math
from typing import Any, Dict, List, Tuple, Callable

import pandas as pd
import matplotlib.pyplot as plt

# mcap_ros2 reader (high-level) is the simplest way to decode ROS 2 messages from MCAP
try:
    from mcap_ros2.reader import read_ros2_messages
except Exception as e:
    raise ImportError(
        "Cannot import mcap_ros2.reader.read_ros2_messages."
        "Please install dependencies: pip install mcap mcap-ros2-support\n"
        f"Original error: {e}"
    )


# ------------------------------
# Helpers
# ------------------------------

def parse_field_path(root: Any, path: str) -> Any:
    """
    Resolve a nested field path like 'angular/x' or 'rigidbodies[3]/pose/position/x'
    against a ROS 2 message object.

    Supports attribute access with '/', and indexing with '[i]' on sequences.
    """
    cur = root
    # split by '/'; ignore leading/trailing slashes
    for seg in [s for s in path.split('/') if s]:
        # handle name[index] possibly repeated (e.g., foo[2][1])
        while True:
            if '[' in seg and seg.endswith(']'):
                name, idx_part = seg.split('[', 1)
                idx_str = idx_part[:-1]  # strip trailing ']'
                if name:
                    cur = getattr(cur, name)
                # index may be int; if empty, error
                idx = int(idx_str)
                cur = cur[idx]
                # after one index handled, check if more remain like [k][m]
                # but since seg endswith ']', we may have chained indices.
                # We'll loop while the last char is ']' and there's another '[' inside.
                # To support chained, we'd need to parse repeatedly; simpler approach:
                # try to find additional [..] patterns:
                # However, since we split only once above, handle multiple with a simple loop:
                # Not implemented for multiple chained brackets beyond one layer.
                break
            else:
                # simple attribute
                cur = getattr(cur, seg)
                break
    return cur


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ns_to_s(ns: int) -> float:
    return ns * 1e-9


# ------------------------------
# Extraction logic
# ------------------------------

def extract(mcap_path: str, out_dir: str):
    ensure_dir(out_dir)

    # Define which fields we want from which topics
    # Mapping: topic -> list of (output_column_name, field_path_in_msg)
    targets: Dict[str, List[Tuple[str, str]]] = {
        "/cmd_vel": [
            ("angular_x", "angular/x"),
            ("angular_y", "angular/y"),
            ("angular_z", "angular/z"),
            ("linear_x",  "linear/x"),
            ("linear_y",  "linear/y"),
            ("linear_z",  "linear/z"),
        ],
        "/rigid_bodies": [
            ("pos_x_rb3", "rigidbodies[3]/pose/position/x"),
            ("pos_y_rb3", "rigidbodies[3]/pose/position/y"),
            ("pos_z_rb3", "rigidbodies[3]/pose/position/z"),
        ],
        "/state_broadcaster/joints_state": [
            ("vel_10", "velocity[10]"),
            ("vel_11", "velocity[11]"),
            ("vel_8",  "velocity[8]"),
            ("vel_9",  "velocity[9]"),
        ],
    }

    # Buffers per topic: list of dict rows including 't' and fields
    data_buffers: Dict[str, List[Dict[str, float]]] = {t: [] for t in targets.keys()}

    first_time_ns: int | None = None

    with open(mcap_path, "rb") as f:
        for msg in read_ros2_messages(f):
            topic = getattr(msg, "topic", None) or getattr(getattr(msg, "channel", None), "topic", None)
            if topic not in targets:
                continue

            log_time_ns = msg.log_time  # int (ns)
            if first_time_ns is None:
                first_time_ns = log_time_ns
            t_rel = ns_to_s(log_time_ns - first_time_ns)

            ros_msg = msg.ros_msg
            row: Dict[str, float] = {"t": t_rel}

            for col_name, field_path in targets[topic]:
                try:
                    value = parse_field_path(ros_msg, field_path)
                    # Ensure numeric (float). Some fields may be numpy types; cast.
                    row[col_name] = float(value)
                except Exception as e:
                    # Leave NaN if missing; this avoids crashing when the array index doesn't exist
                    row[col_name] = math.nan

            data_buffers[topic].append(row)

    # Convert to DataFrames and save CSVs
    dfs: Dict[str, pd.DataFrame] = {}
    for topic, rows in data_buffers.items():
        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
        dfs[topic] = df

    # Write CSVs & create plots
    if "/cmd_vel" in dfs:
        csv_path = os.path.join(out_dir, "cmd_vel.csv")
        dfs["/cmd_vel"].to_csv(csv_path, index=False)

        # Plot angular and linear components in one figure (two subplots -> but requirement says one chart per plot).
        # Therefore, make two separate figures: cmd_vel_angular.png and cmd_vel_linear.png
        df = dfs["/cmd_vel"]

        plt.figure()
        plt.plot(df["t"], df["angular_x"], label="angular_x")
        plt.plot(df["t"], df["angular_y"], label="angular_y")
        plt.plot(df["t"], df["angular_z"], label="angular_z")
        plt.xlabel("time [s]")
        plt.ylabel("angular [rad/s]")
        plt.title("/cmd_vel angular")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "cmd_vel_angular.png"), dpi=150)
        plt.close()

        plt.figure()
        plt.plot(df["t"], df["linear_x"], label="linear_x")
        plt.plot(df["t"], df["linear_y"], label="linear_y")
        plt.plot(df["t"], df["linear_z"], label="linear_z")
        plt.xlabel("time [s]")
        plt.ylabel("linear [m/s]")
        plt.title("/cmd_vel linear")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "cmd_vel_linear.png"), dpi=150)
        plt.close()

    if "/rigid_bodies" in dfs:
        csv_path = os.path.join(out_dir, "rigid_body_pos.csv")
        dfs["/rigid_bodies"].to_csv(csv_path, index=False)

        df = dfs["/rigid_bodies"]
        plt.figure()
        plt.plot(df["t"], df["pos_x_rb3"], label="x")
        plt.plot(df["t"], df["pos_y_rb3"], label="y")
        plt.plot(df["t"], df["pos_z_rb3"], label="z")
        plt.xlabel("time [s]")
        plt.ylabel("position [m]")
        plt.title("Rigid body[3] position")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rigid_body_pos.png"), dpi=150)
        plt.close()

    if "/state_broadcaster/joints_state" in dfs:
        csv_path = os.path.join(out_dir, "joint_velocities.csv")
        dfs["/state_broadcaster/joints_state"].to_csv(csv_path, index=False)

        df = dfs["/state_broadcaster/joints_state"]
        plt.figure()
        plt.plot(df["t"], df["vel_8"],  label="vel[8]")
        plt.plot(df["t"], df["vel_9"],  label="vel[9]")
        plt.plot(df["t"], df["vel_10"], label="vel[10]")
        plt.plot(df["t"], df["vel_11"], label="vel[11]")
        plt.xlabel("time [s]")
        plt.ylabel("joint velocity [rad/s]")
        plt.title("Joint velocities (selected indices)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "joint_velocities.png"), dpi=150)
        plt.close()

    # Summary
    print("Done.")
    for topic in ["/cmd_vel", "/rigid_bodies", "/state_broadcaster/joints_state"]:
        if topic in dfs:
            print(f"- {topic}: {len(dfs[topic])} samples")
        else:
            print(f"- {topic}: not found in MCAP or zero messages")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True, help="Path to the .mcap file")
    ap.add_argument("--out", required=True, help="Output directory for CSVs and plots")
    args = ap.parse_args()

    extract(args.mcap, args.out)


if __name__ == "__main__":
    main()
