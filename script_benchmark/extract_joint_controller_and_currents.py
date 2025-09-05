#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract joint controller commands (positions, velocities) and joint currents from an MCAP file.

Topics:
- /joint_controller/command/position[0..7]
- /joint_controller/command/velocity[8,9,10,11]
- /state_broadcaster/joints_state/current[0..11]

Usage:
    python extract_joint_controller_and_currents.py --mcap path/to/file.mcap --out ./out_dir
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Optional, List, Dict, Tuple

try:
    from mcap_ros2.reader import read_ros2_messages
except Exception as e:
    raise ImportError(
        "Cannot import mcap_ros2.reader.read_ros2_messages. "
        "Install: pip install mcap mcap-ros2-support\n"
        f"Original error: {e}"
    )


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
            return int(v)
    return None

def get_ros_msg(msg):
    return getattr(msg, "ros_msg", None) or getattr(msg, "message", None)

def try_get_scalar(ros_msg: Any) -> Optional[float]:
    for attr in ("data", "value"):
        v = getattr(ros_msg, attr, None)
        if v is not None:
            return float(v)
    return None


def extract(mcap_path: str, out_dir: str):
    ensure_dir(out_dir)

    # topics to collect
    topics = []
    topics += [f"/joint_controller/command/position[{i}]" for i in range(8)]
    topics += [f"/joint_controller/command/velocity[{i}]" for i in (8, 9, 10, 11)]
    topics += [f"/state_broadcaster/joints_state/current[{i}]" for i in range(12)]

    buffers: Dict[str, List[Tuple[float, float]]] = {t: [] for t in topics}
    first_time_ns: Optional[int] = None

    with open(mcap_path, "rb") as f:
        for msg in read_ros2_messages(f):
            topic = get_topic(msg)
            if topic not in topics:
                continue
            t_ns = get_log_time_ns(msg)
            if t_ns is None:
                continue
            if first_time_ns is None:
                first_time_ns = t_ns
            t = ns_to_s(t_ns - first_time_ns)
            ros_msg = get_ros_msg(msg)
            if ros_msg is None:
                continue
            val = try_get_scalar(ros_msg)
            if val is None:
                continue
            buffers[topic].append((t, val))

    # save each topic to csv
    for topic, data in buffers.items():
        if not data:
            continue
        df = pd.DataFrame(data, columns=["t", "value"])
        safe_name = topic.replace("/", "_").replace("[", "_").replace("]", "")
        out_csv = os.path.join(out_dir, f"{safe_name}.csv")
        df.to_csv(out_csv, index=False)

        # also plot
        plt.figure()
        plt.plot(df["t"], df["value"], label=topic)
        plt.xlabel("time [s]"); plt.ylabel("value"); plt.title(topic)
        plt.legend(); plt.tight_layout()
        out_png = out_csv.replace(".csv", ".png")
        plt.savefig(out_png, dpi=150)
        plt.close()

    print("Done. Data saved in:", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True, help="Path to .mcap file")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()
    extract(args.mcap, args.out)


if __name__ == "__main__":
    main()
