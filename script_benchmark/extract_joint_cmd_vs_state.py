#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracts joint command/state arrays from an MCAP, stores them in separate subfolders,
and plots position command vs position state for corresponding joints (by index order).

Folders created inside --out:
  - joint_command/          (CSV+PNG per index for command arrays)
  - state_broadcaster/      (CSV+PNG per index for state arrays)
  - compare/                (PNG per joint: position command vs state)

Usage:
  python extract_joint_cmd_vs_state.py --mcap path/to/file.mcap --out ./out_dir

Requires:
  pip install mcap mcap-ros2-support pandas matplotlib
"""
import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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

def to_float_list(seq) -> Optional[List[float]]:
    if seq is None:
        return None
    try:
        return [float(x) for x in seq]
    except Exception:
        return None

def safe_name(label: str) -> str:
    return label.replace("/", "_").replace("[", "_").replace("]", "")

def save_series(out_dir: str, label: str, t: List[float], v: List[float], ylabel="value"):
    ensure_dir(out_dir)
    fname = safe_name(label)
    csv_path = os.path.join(out_dir, f"{fname}.csv")
    df = pd.DataFrame({"t": t, "value": v})
    df.to_csv(csv_path, index=False)

    plt.figure()
    plt.plot(t, v, label=label)
    plt.xlabel("time [s]"); plt.ylabel(ylabel); plt.title(label); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{fname}.png"), dpi=150)
    plt.close()
    return csv_path

# ---------- main ----------
def extract(mcap_path: str, out_dir: str):
    # Base topics
    topic_cmd = "/joint_controller/command"
    topic_state = "/state_broadcaster/joints_state"

    # Buffers (time series by index)
    # We'll collect arrays per sample and later split into per-index series.
    cmd_time: List[float] = []
    cmd_names_sample: Optional[List[str]] = None
    cmd_pos_samples: List[List[float]] = []     # position[]
    cmd_vel_samples: List[List[float]] = []     # velocity[] (if present)
    cmd_eff_samples: List[List[float]] = []     # effort[] (if present)

    st_time: List[float] = []
    st_names_sample: Optional[List[str]] = None
    st_pos_samples: List[List[float]] = []      # position[]
    st_vel_samples: List[List[float]] = []      # velocity[] (if present)
    st_cur_samples: List[List[float]] = []      # current[] (if present)
    st_eff_samples: List[List[float]] = []      # effort[] (if present)

    first_ns: Optional[int] = None

    with open(mcap_path, "rb") as f:
        for msg in read_ros2_messages(f):
            topic = get_topic(msg)
            if topic not in (topic_cmd, topic_state):
                continue

            ns = get_log_time_ns(msg)
            if ns is None:
                continue
            if first_ns is None:
                first_ns = ns
            t = ns_to_s(ns - first_ns)

            ros_msg = get_ros_msg(msg)
            if ros_msg is None:
                continue

            if topic == topic_cmd:
                # Expected fields by your screenshots:
                # command.name[], command.position[], command.velocity[], command.effort[] ...
                cmd = getattr(ros_msg, "command", None) or ros_msg

                # names (strings)
                if cmd_names_sample is None:
                    names = getattr(cmd, "name", None)
                    if names is not None:
                        cmd_names_sample = [str(x) for x in names]

                # numeric arrays
                pos = to_float_list(getattr(cmd, "position", None))
                vel = to_float_list(getattr(cmd, "velocity", None))
                eff = to_float_list(getattr(cmd, "effort", None))

                # time stamp only if at least one numeric array present
                if pos is not None or vel is not None or eff is not None:
                    cmd_time.append(t)
                    cmd_pos_samples.append(pos or [])
                    cmd_vel_samples.append(vel or [])
                    cmd_eff_samples.append(eff or [])

            elif topic == topic_state:
                # Expected fields: joints_state.name[], position[], velocity[], effort[], current[]
                st = getattr(ros_msg, "joints_state", None) or ros_msg

                if st_names_sample is None:
                    names = getattr(st, "name", None)
                    if names is not None:
                        st_names_sample = [str(x) for x in names]

                pos = to_float_list(getattr(st, "position", None))
                vel = to_float_list(getattr(st, "velocity", None))
                eff = to_float_list(getattr(st, "effort", None))
                cur = to_float_list(getattr(st, "current", None))

                if pos is not None or vel is not None or eff is not None or cur is not None:
                    st_time.append(t)
                    st_pos_samples.append(pos or [])
                    st_vel_samples.append(vel or [])
                    st_eff_samples.append(eff or [])
                    st_cur_samples.append(cur or [])

    # Convert stacked samples -> per-index series
    out_cmd = os.path.join(out_dir, "joint_command")
    out_st  = os.path.join(out_dir, "state_broadcaster")
    out_cmp = os.path.join(out_dir, "compare")
    ensure_dir(out_cmd); ensure_dir(out_st); ensure_dir(out_cmp)

    # Save name orders for transparency
    if cmd_names_sample:
        pd.DataFrame({"index": list(range(len(cmd_names_sample))), "name": cmd_names_sample}) \
          .to_csv(os.path.join(out_cmd, "name_order.csv"), index=False)
    if st_names_sample:
        pd.DataFrame({"index": list(range(len(st_names_sample))), "name": st_names_sample}) \
          .to_csv(os.path.join(out_st, "name_order.csv"), index=False)

    # Helper to extract a single index column safely from stacked samples
    def index_series(times: List[float], stacked: List[List[float]], idx: int) -> Tuple[List[float], List[float]]:
        ts, vals = [], []
        for t, arr in zip(times, stacked):
            if idx < len(arr):
                ts.append(t)
                vals.append(arr[idx])
        return ts, vals

    # Determine how many joints to compare (in the mulinex robot, 12)
    n_cmd = len(cmd_names_sample) if cmd_names_sample is not None else 0
    n_st  = len(st_names_sample)  if st_names_sample  is not None else 0
    n = max(n_cmd, n_st)

    # Save per-index series for command and state (position & optional others)
    for i in range(n):
        # command side
        if cmd_time:
            t_cmd_pos, v_cmd_pos = index_series(cmd_time, cmd_pos_samples, i)
            if t_cmd_pos:
                label = f"/joint_controller/command/position[{i}]" if (cmd_names_sample is None) else f"/joint_controller/command/position[{i}]_{cmd_names_sample[i] if i<n_cmd else 'UNK'}"
                save_series(out_cmd, label, t_cmd_pos, v_cmd_pos, ylabel="position [rad]")

            t_cmd_vel, v_cmd_vel = index_series(cmd_time, cmd_vel_samples, i)
            if t_cmd_vel:
                label = f"/joint_controller/command/velocity[{i}]" if (cmd_names_sample is None) else f"/joint_controller/command/velocity[{i}]_{cmd_names_sample[i] if i<n_cmd else 'UNK'}"
                save_series(out_cmd, label, t_cmd_vel, v_cmd_vel, ylabel="velocity [rad/s]")

            t_cmd_eff, v_cmd_eff = index_series(cmd_time, cmd_eff_samples, i)
            if t_cmd_eff:
                label = f"/joint_controller/command/effort[{i}]" if (cmd_names_sample is None) else f"/joint_controller/command/effort[{i}]_{cmd_names_sample[i] if i<n_cmd else 'UNK'}"
                save_series(out_cmd, label, t_cmd_eff, v_cmd_eff, ylabel="effort")

        # state side
        if st_time:
            t_st_pos, v_st_pos = index_series(st_time, st_pos_samples, i)
            if t_st_pos:
                label = f"/state_broadcaster/joints_state/position[{i}]" if (st_names_sample is None) else f"/state_broadcaster/joints_state/position[{i}]_{st_names_sample[i] if i<n_st else 'UNK'}"
                save_series(out_st, label, t_st_pos, v_st_pos, ylabel="position [rad]")

            t_st_vel, v_st_vel = index_series(st_time, st_vel_samples, i)
            if t_st_vel:
                label = f"/state_broadcaster/joints_state/velocity[{i}]" if (st_names_sample is None) else f"/state_broadcaster/joints_state/velocity[{i}]_{st_names_sample[i] if i<n_st else 'UNK'}"
                save_series(out_st, label, t_st_vel, v_st_vel, ylabel="velocity [rad/s]")

            t_st_eff, v_st_eff = index_series(st_time, st_eff_samples, i)
            if t_st_eff:
                label = f"/state_broadcaster/joints_state/effort[{i}]"
                save_series(out_st, label, t_st_eff, v_st_eff, ylabel="effort")

            t_st_cur, v_st_cur = index_series(st_time, st_cur_samples, i)
            if t_st_cur:
                label = f"/state_broadcaster/joints_state/current[{i}]"
                save_series(out_st, label, t_st_cur, v_st_cur, ylabel="current [A]")

    # -------- comparison plots: position command vs position state ----------
    # Use index correspondence (0↔0, 1↔1, ...), and print a small warning if names differ.
    for i in range(n):
        name_cmd = cmd_names_sample[i] if (cmd_names_sample and i < n_cmd) else f"idx{i}"
        name_st  = st_names_sample[i]  if (st_names_sample  and i < n_st)  else f"idx{i}"
        if cmd_names_sample and st_names_sample and i < n_cmd and i < n_st and (name_cmd != name_st):
            print(f"[WARN] Name mismatch at index {i}: command='{name_cmd}' vs state='{name_st}'. Using index pairing.")

        # fetch position series
        t_cmd_pos, v_cmd_pos = index_series(cmd_time, cmd_pos_samples, i)
        t_st_pos,  v_st_pos  = index_series(st_time,  st_pos_samples,  i)

        if not t_cmd_pos or not t_st_pos:
            # not enough data to compare
            continue

        # merge-asof on time to overlay properly
        df_cmd = pd.DataFrame({"t": t_cmd_pos, "pos_cmd": v_cmd_pos}).sort_values("t")
        df_st  = pd.DataFrame({"t": t_st_pos,  "pos_st":  v_st_pos}).sort_values("t")
        df = pd.merge_asof(df_cmd, df_st, on="t", direction="nearest")

        plt.figure()
        plt.plot(df["t"], df["pos_cmd"], label=f"command ({name_cmd})")
        plt.plot(df["t"], df["pos_st"],  label=f"state   ({name_st})")
        plt.xlabel("time [s]"); plt.ylabel("position [rad]")
        plt.title(f"Position: command vs state – index {i} ({name_cmd})")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_cmp, f"compare_position_idx{i}_{safe_name(name_cmd)}.png"), dpi=150)
        plt.close()

    print(f"Done.\n- Saved per-index series in:\n  {out_cmd}\n  {out_st}\n- Comparison plots in:\n  {out_cmp}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True, help="Path to .mcap file")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()
    extract(args.mcap, args.out)

if __name__ == "__main__":
    main()
