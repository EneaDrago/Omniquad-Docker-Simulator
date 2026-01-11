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

def detect_motion_interval(t: np.ndarray, series: np.ndarray,
                           on_threshold: float = 0.10,
                           baseline: Optional[float] = None,
                           baseline_samples: int = 5) -> Tuple[Optional[float], Optional[float]]:
    """
    Detect start/end of motion comparing deviations from a resting baseline.

    - `baseline`: if provided, used as resting value; otherwise computed as the
      median of the first `baseline_samples` entries of `series` (or first entry
      if fewer samples are available).
    - `on_threshold`: threshold for |series - baseline| to consider motion.

    Returns (t_start, t_end) in seconds (same units as `t`). If not found,
    returns (None, None).
    """
    if len(t) == 0 or len(series) == 0:
        return (None, None)

    # compute baseline from first samples if not provided
    if baseline is None:
        n = min(max(1, baseline_samples), len(series))
        try:
            baseline = float(np.nanmedian(series[:n]))
        except Exception:
            baseline = float(series[0])

    # detect where the series deviates from baseline by more than threshold
    mask_on = np.abs(series - baseline) > on_threshold
    if not mask_on.any():
        return (None, None)

    start_idx = int(np.argmax(mask_on))
    end_indices = np.where(mask_on)[0]
    end_idx = int(end_indices[-1]) if len(end_indices) > 0 else None

    t_start = float(t[start_idx])
    t_end = float(t[end_idx]) if end_idx is not None else None
    return (t_start, t_end)

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
    # Match by joint NAME (not index) to handle different orderings
    
    # Build name-to-index mappings
    cmd_name_to_idx = {}
    if cmd_names_sample:
        for idx, name in enumerate(cmd_names_sample):
            cmd_name_to_idx[name] = idx
    
    st_name_to_idx = {}
    if st_names_sample:
        for idx, name in enumerate(st_names_sample):
            st_name_to_idx[name] = idx
    
    # Find common joint names
    cmd_names_set = set(cmd_name_to_idx.keys())
    st_names_set = set(st_name_to_idx.keys())
    common_names = sorted(cmd_names_set & st_names_set)
    
    # If no common names, fall back to index-based pairing
    if not common_names:
        print("[INFO] No common joint names found between command and state. Using index-based pairing.")
        common_names = [f"idx{i}" for i in range(n)]
        use_index_pairing = True
    else:
        use_index_pairing = False
        print(f"[INFO] Found {len(common_names)} common joints: {common_names}")
    
    # --- Compute a single global motion interval for this file (per iteration) ---
    t_start_global: Optional[float] = None
    t_end_global: Optional[float] = None
    if st_time and st_vel_samples:
        # Build a reference scalar velocity series by averaging absolute joint velocities
        try:
            t_st = np.array(st_time)
            vel_ref = np.array([np.nan if (not arr) else np.nanmean(np.abs(arr)) for arr in st_vel_samples], dtype=float)
            # drop nan pairs
            valid = ~np.isnan(vel_ref)
            if valid.any():
                t_st_valid = t_st[valid]
                vel_ref_valid = vel_ref[valid]
                # baseline as median of first few samples
                baseline_n = min(5, len(vel_ref_valid))
                baseline_val = float(np.nanmedian(vel_ref_valid[:baseline_n]))
                t_start_global, t_end_global = detect_motion_interval(t_st_valid, vel_ref_valid,
                                                                     on_threshold=0.10,
                                                                     baseline=baseline_val,
                                                                     baseline_samples=baseline_n)
                print(f"[INFO] Global motion interval: t_start={t_start_global}, t_end={t_end_global}")
        except Exception:
            t_start_global, t_end_global = (None, None)
    
    # Plot comparison for each joint
    joint_errors = []  # lista di (joint_name, mae_pos, t_start, t_end)
    
    for joint_name in common_names:
        if use_index_pairing:
            # Fallback: use index
            i = int(joint_name.replace("idx", ""))
            idx_cmd = i if i < n_cmd else None
            idx_st = i if i < n_st else None
        else:
            # Use name-based matching
            idx_cmd = cmd_name_to_idx.get(joint_name)
            idx_st = st_name_to_idx.get(joint_name)
        
        if idx_cmd is None or idx_st is None:
            continue
        
        # fetch position series
        t_cmd_pos, v_cmd_pos = index_series(cmd_time, cmd_pos_samples, idx_cmd)
        t_st_pos,  v_st_pos  = index_series(st_time,  st_pos_samples,  idx_st)

        if not t_cmd_pos or not t_st_pos:
            # not enough data to compare
            continue

        # merge-asof on time to overlay properly
        df_cmd = pd.DataFrame({"t": t_cmd_pos, "pos_cmd": v_cmd_pos}).sort_values("t")
        df_st  = pd.DataFrame({"t": t_st_pos,  "pos_st":  v_st_pos}).sort_values("t")
        df = pd.merge_asof(df_cmd, df_st, on="t", direction="nearest")

        # Decide whether to compare position or velocity (wheels are velocity-controlled)
        is_wheel = ("WHEEL" in joint_name.upper())

        if is_wheel:
            # compare velocities
            t_cmd_v, v_cmd_v = index_series(cmd_time, cmd_vel_samples, idx_cmd)
            t_st_v,  v_st_v  = index_series(st_time,  st_vel_samples,  idx_st)
            if not t_cmd_v or not t_st_v:
                # fallback to position if velocities missing
                compare_type = "position"
            else:
                compare_type = "velocity"
        else:
            compare_type = "position"

        if compare_type == "position":
            df_cmd = pd.DataFrame({"t": t_cmd_pos, "val_cmd": v_cmd_pos}).sort_values("t")
            df_st  = pd.DataFrame({"t": t_st_pos,  "val_st":  v_st_pos}).sort_values("t")
        else:
            df_cmd = pd.DataFrame({"t": t_cmd_v, "val_cmd": v_cmd_v}).sort_values("t")
            df_st  = pd.DataFrame({"t": t_st_v,  "val_st":  v_st_v}).sort_values("t")

        df = pd.merge_asof(df_cmd, df_st, on="t", direction="nearest")
        df["err"] = df["val_cmd"] - df["val_st"]

        # Use the global motion interval if available; otherwise fallback to per-joint detection
        if t_start_global is not None and t_end_global is not None:
            t_start = t_start_global
            t_end = t_end_global
        else:
            # fallback: detect per-joint using velocity derived from state values
            t_arr = df["t"].to_numpy()
            try:
                vel_st = np.gradient(df["val_st"].to_numpy(), t_arr)
                t_start, t_end = detect_motion_interval(t_arr, vel_st, on_threshold=0.10)
            except Exception:
                t_start, t_end = (None, None)

        # Calcola MAE durante il movimento
        if t_start is not None and t_end is not None:
            mask = (df["t"] >= t_start) & (df["t"] <= t_end)
            mae = float(np.nanmean(np.abs(df.loc[mask, "err"])))
        else:
            mae = float(np.nanmean(np.abs(df["err"])))

        joint_errors.append((joint_name, mae, t_start, t_end, compare_type))
        print(f"  {joint_name}: MAE = {mae:.6f} ({compare_type})")

        # Save timeseries of error for later aggregation
        timeseries_dir = os.path.join(out_dir, "compare_timeseries")
        ensure_dir(timeseries_dir)
        ts_csv = os.path.join(timeseries_dir, f"err_{safe_name(joint_name)}.csv")
        pd.DataFrame({"t": df["t"], "err": df["err"]}).to_csv(ts_csv, index=False)

        # Plot command vs state
        plt.figure()
        plt.plot(df["t"], df["val_cmd"], label=f"command ({joint_name})")
        plt.plot(df["t"], df["val_st"],  label=f"state   ({joint_name})")
        if t_start is not None and t_end is not None:
            plt.axvline(t_start, linestyle="--", color="red", linewidth=2, alpha=0.5)
            plt.axvline(t_end, linestyle="--", color="red", linewidth=2, alpha=0.5)
        plt.xlabel("time [s]")
        plt.ylabel("velocity [rad/s]" if compare_type=="velocity" else "position [rad]")
        ctrl_label = 'Velocity control' if compare_type == 'velocity' else 'Position control'
        plt.title(f"{ctrl_label}: command vs state - {joint_name}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_cmp, f"compare_{compare_type}_{safe_name(joint_name)}.png"), dpi=150)
        plt.close()
    
    # Salva errori joint in CSV
    if joint_errors:
        df_errors = pd.DataFrame(joint_errors, columns=["joint_name", "mae_pos", "t_start", "t_end", "compare_type"])
        errors_csv = os.path.join(out_dir, "joint_error_summary.csv")
        df_errors.to_csv(errors_csv, index=False)
        print(f"\nJoint errors saved in: {errors_csv}")

    print(f"Done.\n- Saved per-index series in:\n  {out_cmd}\n  {out_st}\n- Comparison plots in:\n  {out_cmp}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True, help="Path to .mcap file")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()
    extract(args.mcap, args.out)

if __name__ == "__main__":
    main()
