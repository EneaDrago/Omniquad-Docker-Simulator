#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract joint controller commands (positions, velocities) and joint currents from an MCAP file.

It supports these common layouts:
- /joint_controller/command/position  -> message has array: .position or .data
- /joint_controller/command/velocity  -> message has array: .velocity or .data
- /state_broadcaster/joints_state     -> message has array: .current (custom) or .effort, etc.

For each requested index, saves a CSV (t,value) and a PNG plot.

Usage:
  # list topics only
  python extract_joint_controller_and_currents.py --mcap path/to/file.mcap --list

  # extract
  python extract_joint_controller_and_currents.py --mcap path/to/file.mcap --out ./out_dir
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from mcap_ros2.reader import read_ros2_messages
except Exception as e:
    raise ImportError(
        "Cannot import mcap_ros2.reader.read_ros2_messages. "
        "Install: pip install mcap mcap-ros2-support\n"
        f"Original error: {e}"
    )

# ---------- utils ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ns_to_s(ns: int) -> float:
    return ns * 1e-9

def get_topic(msg) -> Optional[str]:
    return getattr(msg, "topic", None) or getattr(getattr(msg, "channel", None), "topic", None)

def get_log_time_ns(msg) -> Optional[int]:
    for attr in ("log_time", "log_time_ns", "publish_time"):
        v = getattr(msg, attr, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return None

def get_ros_msg(msg):
    return getattr(msg, "ros_msg", None) or getattr(msg, "message", None)

def get_array_from_msg(ros_msg: Any, candidates=("position","velocity","current","effort","data")) -> Optional[List[float]]:
    """Return first array-like attribute found among candidates."""
    for name in candidates:
        arr = getattr(ros_msg, name, None)
        if arr is not None:
            try:
                return list(map(float, arr))
            except Exception:
                # might be non-iterable or not numeric
                continue
    return None

def save_series(out_dir: str, label: str, t: List[float], v: List[float], ylabel="value"):
    safe = label.replace("/", "_").replace("[", "_").replace("]", "")
    csv_path = os.path.join(out_dir, f"{safe}.csv")
    df = pd.DataFrame({"t": t, "value": v})
    df.to_csv(csv_path, index=False)

    plt.figure()
    plt.plot(t, v, label=label)
    plt.xlabel("time [s]"); plt.ylabel(ylabel); plt.title(label); plt.legend(); plt.tight_layout()
    plt.savefig(csv_path.replace(".csv", ".png"), dpi=150)
    plt.close()

# ---------- main extraction ----------
def extract(mcap_path: str, out_dir: str):
    ensure_dir(out_dir)

    # What we want to export (indices per topic/array field)
    # You asked for:
    # - /joint_controller/command/position[0..7]
    # - /joint_controller/command/velocity[8,9,10,11]
    # - /state_broadcaster/joints_state/current[0..11]
    #
    # In MCAP these will be base topics with arrays; we’ll read arrays and pick elements.
    wanted = {
        "/joint_controller/command/position": list(range(0, 8)),
        "/joint_controller/command/velocity": [8, 9, 10, 11],
        "/state_broadcaster/joints_state": list(range(0, 12)),  # pulling .current[] if available
    }

    # Buffers: label -> list of (t, value)
    buffers: Dict[str, List[Tuple[float, float]]] = {}
    first_time_ns: Optional[int] = None

    # Also allow some aliases: sometimes the command topics are named differently
    aliases = {
        "/joint_controller/command/pos": "/joint_controller/command/position",
        "/joint_controller/command/vel": "/joint_controller/command/velocity",
        # if your bag uses a single "/joint_controller/command" with arrays 'position' and 'velocity'
        "/joint_controller/command": "/joint_controller/command",  # handled specially below
    }

    with open(mcap_path, "rb") as f:
        for msg in read_ros2_messages(f):
            topic = get_topic(msg)
            if topic is None:
                continue

            # direct match or alias
            base_topic = aliases.get(topic, topic)

            t_ns = get_log_time_ns(msg)
            if t_ns is None:
                continue
            if first_time_ns is None:
                first_time_ns = t_ns
            t = ns_to_s(t_ns - first_time_ns)

            ros_msg = get_ros_msg(msg)
            if ros_msg is None:
                continue

            # Handle /joint_controller/command that might contain both position/velocity arrays
            if base_topic == "/joint_controller/command":
                # Try both arrays if present
                for subname, out_topic, idx_list in [
                    ("position", "/joint_controller/command/position", wanted.get("/joint_controller/command/position", [])),
                    ("velocity", "/joint_controller/command/velocity", wanted.get("/joint_controller/command/velocity", [])),
                ]:
                    arr = getattr(ros_msg, subname, None) or getattr(ros_msg, "data", None)
                    if arr is None:
                        continue
                    arr_list = list(arr)
                    for idx in idx_list:
                        if idx < len(arr_list):
                            label = f"{out_topic}[{idx}]"
                            buffers.setdefault(label, []).append((t, float(arr_list[idx])))
                continue

            # Normal case: base topic present in 'wanted'
            if base_topic not in wanted:
                continue

            arr = get_array_from_msg(ros_msg)
            if arr is None:
                # Nothing usable in this message
                continue

            for idx in wanted[base_topic]:
                if idx < len(arr):
                    label = f"{base_topic}[{idx}]"
                    buffers.setdefault(label, []).append((t, float(arr[idx])))

    # Save files
    for label, rows in buffers.items():
        if not rows:
            continue
        rows.sort(key=lambda x: x[0])
        t = [r[0] for r in rows]
        v = [r[1] for r in rows]
        # ylabel per family
        if label.startswith("/state_broadcaster/joints_state"):
            ylabel = "current [A]"
        elif "/velocity" in label:
            ylabel = "velocity [rad/s]"
        else:
            ylabel = "position [rad]"
        save_series(out_dir, label, t, v, ylabel=ylabel)

    # quick summary
    print("Done. Saved series:")
    for label, rows in buffers.items():
        print(f"  {label}: {len(rows)} samples -> {label.replace('/', '_').replace('[', '_').replace(']', '')}.csv")

def list_topics(mcap_path: str):
    seen = {}
    with open(mcap_path, "rb") as f:
        for msg in read_ros2_messages(f):
            topic = get_topic(msg)
            if topic:
                seen[topic] = seen.get(topic, 0) + 1
    if not seen:
        print("No topics found.")
    else:
        print("Topics in MCAP:")
        for k, v in sorted(seen.items()):
            print(f"{k}  ({v} msgs)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True, help="Path to .mcap file")
    ap.add_argument("--out", help="Output directory (required unless --list)")
    ap.add_argument("--list", action="store_true", help="List topics in the MCAP and exit")
    args = ap.parse_args()

    if args.list:
        list_topics(args.mcap)
        return
    if not args.out:
        raise SystemExit("Please provide --out or use --list")
    extract(args.mcap, args.out)

if __name__ == "__main__":
    main()
