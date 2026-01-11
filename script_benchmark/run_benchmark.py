#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parent script that runs benchmarks for a given test.

Usage:
        python run_benchmark.py --test_name avanti
        python run_benchmark.py --test_name discesa_5deg

Expected structure:
    rosbag/REAL_ROBOT_rough_GIOVEDI/{test_name}/benchmark_*_v1/
    rosbag/REAL_ROBOT_rough_GIOVEDI/{test_name}/benchmark_*_v2/
    ...
    rosbag/REAL_ROBOT_rough_GIOVEDI/{test_name}/benchmark_*_v5/

Output saved in:
    script_benchmark/output/{test_name}/
        v1/, v2/, ..., v5/ (each with state_vs_cmd/, state_vs_vicon/, cmd_vs_vicon/)
        unificati/ (aggregated error files)
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_project_root():
    """Ritorna il path della cartella script_benchmark"""
    return Path(__file__).parent

def get_rosbag_dir():
    """Ritorna il path della cartella rosbag/REAL_ROBOT_rough_GIOVEDI"""
    project_root = get_project_root()
    return project_root.parent / "rosbag" / "REAL_ROBOT_rough_GIOVEDI"

def find_mcap_files(version_dir):
    """Trova tutti i file .mcap nella directory della versione"""
    version_path = Path(version_dir)
    if not version_path.exists():
        return []
    mcap_files = list(version_path.glob("**/*.mcap"))
    return sorted(mcap_files)

def find_version_directories(test_rosbag_dir):
    """
    Trova tutte le directory di versione nel formato: benchmark_*_v1, benchmark_*_v2, etc.
    Ritorna un dict: {version_number: path}
    Es: {1: Path('...benchmark_vel_x_pos_poca_vibrazione_v1'), 2: Path('...v2'), ...}
    """
    test_path = Path(test_rosbag_dir)
    if not test_path.exists():
        return {}
    
    version_dirs = {}
    
    # Cerca tutte le directory che contengono il pattern _vN (dove N è un numero)
    for item in sorted(test_path.iterdir()):
        if not item.is_dir():
            continue
        
        dir_name = item.name
        # Cercare il pattern _vN alla fine del nome
        import re
        match = re.search(r'_v(\d+)$', dir_name)
        if match:
            version_num = int(match.group(1))
            version_dirs[version_num] = item
    
    return version_dirs

def run_extraction_scripts(test_name, version_num, mcap_file, output_dir):
    """Runs the three extraction scripts for an mcap file"""
    
    script_dir = get_project_root()
    version_str = f"v{version_num}"
    
    # Script 1: extract_mcap_body_state_vs_vicon.py
    script1 = script_dir / "extract_mcap_body_state_vs_vicon.py"
    output_dir_vicon = output_dir / version_str / "state_vs_vicon"
    output_dir_vicon.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Running: {script1.name}")
    print(f"Input: {mcap_file}")
    print(f"Output: {output_dir_vicon}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script1), "--mcap", str(mcap_file), "--out", str(output_dir_vicon)],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script1.name}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    
    # Script 2: extract_joint_cmd_vs_state.py
    script2 = script_dir / "extract_joint_cmd_vs_state.py"
    output_dir_joint = output_dir / version_str / "state_vs_cmd"
    output_dir_joint.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Running: {script2.name}")
    print(f"Input: {mcap_file}")
    print(f"Output: {output_dir_joint}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script2), "--mcap", str(mcap_file), "--out", str(output_dir_joint)],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script2.name}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    
    # Script 3: extract_mcap_body_cmd_vs_vicon.py
    script3 = script_dir / "extract_mcap_body_cmd_vs_vicon.py"
    output_dir_cmd = output_dir / version_str / "cmd_vs_vicon"
    output_dir_cmd.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Running: {script3.name}")
    print(f"Input: {mcap_file}")
    print(f"Output: {output_dir_cmd}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script3), "--mcap", str(mcap_file), "--out", str(output_dir_cmd)],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script3.name}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    
    return True

def aggregate_errors(test_name, output_base_dir):
    """Aggrega gli errori dalle diverse versioni"""
    
    unificati_dir = output_base_dir / "unificati"
    unificati_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("Aggregating errors from all versions")
    print(f"{'='*70}")
    
    # Read velocity_error_summary.csv files from all versions
    vicon_errors = []
    cmd_errors = []
    vicon_timeseries = {}  # {version: (t_normalized, err_vx)}
    cmd_timeseries = {}    # {version: (t_normalized, err_vx)}
    joint_errors_all = []  # for aggregating joint errors
    
    colors = {
        1: '#1f77b4',  # blu
        2: '#ff7f0e',  # arancione
        3: '#2ca02c',  # verde
        4: '#d62728',  # rosso
        5: '#9467bd'   # viola
    }
    
    for v in range(1, 6):
        version = f"v{v}"
        
        # Errori Vicon - summary
        vicon_csv = output_base_dir / version / "state_vs_vicon" / "velocity_error_summary.csv"
        if vicon_csv.exists():
            df = pd.read_csv(vicon_csv)
            df["version"] = version
            vicon_errors.append(df)
        
        # Errori Vicon - timeseries
        body_velocity_csv = output_base_dir / version / "state_vs_vicon" / "body_velocity_state_vs_vicon.csv"
        if body_velocity_csv.exists():
            df_ts = pd.read_csv(body_velocity_csv)
            
            # Leggi i tempi di start e end
            t_start = df["t_start"].values[0] if vicon_csv.exists() else None
            t_end = df["t_end"].values[0] if vicon_csv.exists() else None
            
            if t_start is not None and t_end is not None and not np.isnan(t_start) and not np.isnan(t_end):
                # Normalizza il tempo: 0 = t_start, 1 = t_end
                t_data = df_ts["t"].values
                t_normalized = (t_data - t_start) / (t_end - t_start)
                err_vx = df_ts["err_vx"].values
                
                # Filtra solo i dati durante il movimento (0 a 1)
                mask = (t_normalized >= 0) & (t_normalized <= 1)
                vicon_timeseries[v] = (t_normalized[mask], err_vx[mask])
            else:
                # Se non hai t_start/t_end, usa il tempo originale
                vicon_timeseries[v] = (df_ts["t"].values, df_ts["err_vx"].values)
        
        # Errori Joint
        joint_csv = output_base_dir / version / "state_vs_cmd" / "joint_error_summary.csv"
        if joint_csv.exists():
            df_joint = pd.read_csv(joint_csv)
            df_joint["version"] = version
            joint_errors_all.append(df_joint)
        
        # Errori Cmd - summary
        cmd_summary_csv = output_base_dir / version / "cmd_vs_vicon" / "velocity_error_summary.csv"
        if cmd_summary_csv.exists():
            df_cmd = pd.read_csv(cmd_summary_csv)
            df_cmd["version"] = version
            cmd_errors.append(df_cmd)
        
        # Errori Cmd - timeseries
        body_velocity_cmd_csv = output_base_dir / version / "cmd_vs_vicon" / "body_velocity_cmd_vs_vicon.csv"
        if body_velocity_cmd_csv.exists():
            df_ts_cmd = pd.read_csv(body_velocity_cmd_csv)
            
            # Read the start and end times
            t_start_cmd = df_cmd["t_start"].values[0] if cmd_summary_csv.exists() else None
            t_end_cmd = df_cmd["t_end"].values[0] if cmd_summary_csv.exists() else None
            
            if t_start_cmd is not None and t_end_cmd is not None and not np.isnan(t_start_cmd) and not np.isnan(t_end_cmd):
                # Normalize time: 0 = t_start, 1 = t_end
                t_data_cmd = df_ts_cmd["t"].values
                t_normalized_cmd = (t_data_cmd - t_start_cmd) / (t_end_cmd - t_start_cmd)
                err_vx_cmd = df_ts_cmd["err_vx"].values
                
                # Filter only motion data (0 to 1)
                mask_cmd = (t_normalized_cmd >= 0) & (t_normalized_cmd <= 1)
                cmd_timeseries[v] = (t_normalized_cmd[mask_cmd], err_vx_cmd[mask_cmd])
            else:
                # If no t_start/t_end, use original time
                cmd_timeseries[v] = (df_ts_cmd["t"].values, df_ts_cmd["err_vx"].values)
    
    # Aggrega errori Vicon - summary
    if vicon_errors:
        df_vicon_all = pd.concat(vicon_errors, ignore_index=True)
        
        # Salva CSV aggregato
        csv_path = unificati_dir / "confronto_errori_vicon.csv"
        df_vicon_all.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        
        # Crea plot aggregato - bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(df_vicon_all["version"], df_vicon_all["mae_vx"])
        plt.xlabel("Version")
        plt.ylabel("MAE v_x [m/s]")
        plt.title(f"Vicon error comparison - Test: {test_name}")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        png_path = unificati_dir / "confronto_errori_vicon.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Saved: {png_path}")
    
    # Crea grafico sovrapposto degli errori nel tempo
    if vicon_timeseries:
        plt.figure(figsize=(14, 7))
        
        for v in sorted(vicon_timeseries.keys()):
            t_norm, err_vx = vicon_timeseries[v]
            plt.plot(t_norm, err_vx, label=f"v{v}", color=colors[v], linewidth=2, alpha=0.8)
        
        # Aggiungi linee verticali di start e end
        plt.axvline(0, linestyle="--", color="red", linewidth=2, alpha=0.5, label="start")
        plt.axvline(1, linestyle="--", color="red", linewidth=2, alpha=0.5, label="end")
        
        plt.xlabel("Normalized time (0=start, 1=end)")
        plt.ylabel("v_x error [m/s]")
        plt.title(f"Vicon error over time - All versions (Test: {test_name})")
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        png_path = unificati_dir / "errore_vicon_timeseries.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Saved: {png_path}")
        
        # Crea grafico della MEDIA degli errori nel tempo
        # Interpolate tutti i dati su una griglia temporale comune
        t_grid = np.linspace(0, 1, 500)
        errors_interp = []
        
        for v in sorted(vicon_timeseries.keys()):
            t_norm, err_vx = vicon_timeseries[v]
            # Interpola i dati su griglia comune
            err_interp = np.interp(t_grid, t_norm, err_vx, left=np.nan, right=np.nan)
            errors_interp.append(err_interp)
        
        # Calcola media e std
        errors_interp = np.array(errors_interp)
        err_mean = np.nanmean(errors_interp, axis=0)
        err_std = np.nanstd(errors_interp, axis=0)
        
        # Plotta media ± std
        plt.figure(figsize=(14, 7))
        plt.plot(t_grid, err_mean, label="Media", color="black", linewidth=3)
        plt.fill_between(t_grid, err_mean - err_std, err_mean + err_std, 
                         alpha=0.3, color="gray", label="±1 std dev")
        
        # Aggiungi linee verticali di start e end
        plt.axvline(0, linestyle="--", color="red", linewidth=2, alpha=0.5, label="start")
        plt.axvline(1, linestyle="--", color="red", linewidth=2, alpha=0.5, label="end")
        
        plt.xlabel("Normalized time (0=start, 1=end)")
        plt.ylabel("v_x error [m/s]")
        plt.title(f"Mean Vicon error - Test: {test_name}")
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        png_path = unificati_dir / "errore_vicon_media.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Saved: {png_path}")
    
    # Aggregate Cmd errors - summary
    if cmd_errors:
        df_cmd_all = pd.concat(cmd_errors, ignore_index=True)
        
        # Save aggregated CSV
        csv_path = unificati_dir / "confronto_errori_cmd.csv"
        df_cmd_all.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        
        # Create aggregated plot - bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(df_cmd_all["version"], df_cmd_all["mae_vx"])
        plt.xlabel("Version")
        plt.ylabel("MAE v_x [m/s]")
        plt.title(f"Command error comparison - Test: {test_name}")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        png_path = unificati_dir / "confronto_errori_cmd.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Saved: {png_path}")
    
    # Create overlapped plot of cmd errors over time
    if cmd_timeseries:
        plt.figure(figsize=(14, 7))
        
        for v in sorted(cmd_timeseries.keys()):
            t_norm, err_vx = cmd_timeseries[v]
            plt.plot(t_norm, err_vx, label=f"v{v}", color=colors[v], linewidth=2, alpha=0.8)
        
        # Add vertical lines for start and end
        plt.axvline(0, linestyle="--", color="red", linewidth=2, alpha=0.5, label="start")
        plt.axvline(1, linestyle="--", color="red", linewidth=2, alpha=0.5, label="end")
        
        plt.xlabel("Normalized time (0=start, 1=end)")
        plt.ylabel("v_x error [m/s]")
        plt.title(f"Command error over time - All versions (Test: {test_name})")
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        png_path = unificati_dir / "errore_cmd_timeseries.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Saved: {png_path}")
        
        # Create plot of MEAN errors over time
        # Interpolate all data on a common time grid
        t_grid = np.linspace(0, 1, 500)
        errors_interp_cmd = []
        
        for v in sorted(cmd_timeseries.keys()):
            t_norm, err_vx = cmd_timeseries[v]
            # Interpolate data on common grid
            err_interp = np.interp(t_grid, t_norm, err_vx, left=np.nan, right=np.nan)
            errors_interp_cmd.append(err_interp)
        
        # Calculate mean and std
        errors_interp_cmd = np.array(errors_interp_cmd)
        err_mean_cmd = np.nanmean(errors_interp_cmd, axis=0)
        err_std_cmd = np.nanstd(errors_interp_cmd, axis=0)
        
        # Plot mean ± std
        plt.figure(figsize=(14, 7))
        plt.plot(t_grid, err_mean_cmd, label="Mean", color="black", linewidth=3)
        plt.fill_between(t_grid, err_mean_cmd - err_std_cmd, err_mean_cmd + err_std_cmd, 
                         alpha=0.3, color="gray", label="±1 std dev")
        
        # Add vertical lines for start and end
        plt.axvline(0, linestyle="--", color="red", linewidth=2, alpha=0.5, label="start")
        plt.axvline(1, linestyle="--", color="red", linewidth=2, alpha=0.5, label="end")
        
        plt.xlabel("Normalized time (0=start, 1=end)")
        plt.ylabel("v_x error [m/s]")
        plt.title(f"Mean command error - Test: {test_name}")
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        png_path = unificati_dir / "errore_cmd_media.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Saved: {png_path}")
    
    # Aggrega errori Joint
    if joint_errors_all:
        df_joint_all = pd.concat(joint_errors_all, ignore_index=True)
        
        # Salva CSV aggregato
        csv_path = unificati_dir / "confronto_errori_joint.csv"
        df_joint_all.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        
        # Crea plot bar chart per ogni joint
        unique_joints = df_joint_all["joint_name"].unique()
        
        for joint_name in sorted(unique_joints):
            df_joint_filtered = df_joint_all[df_joint_all["joint_name"] == joint_name]
            
            plt.figure(figsize=(10, 6))
            plt.bar(df_joint_filtered["version"], df_joint_filtered["mae_pos"])
            # decide units and title based on compare_type
            compare_types = df_joint_filtered['compare_type'].dropna().unique()
            compare_type = compare_types[0] if len(compare_types) > 0 else 'position'
            if compare_type == 'velocity':
                ylabel = 'MAE [rad/s]'
                ctrl_title = ' (velocity control)'
            else:
                ylabel = 'MAE [rad]'
                ctrl_title = ' (position control)'

            plt.xlabel("Version")
            plt.ylabel(ylabel)
            plt.title(f"Joint error comparison - {joint_name}{ctrl_title}")
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()

            png_path = unificati_dir / f"confronto_errori_joint_{safe_name(joint_name)}.png"
            plt.savefig(png_path, dpi=150)
            plt.close()
            print(f"Saved: {png_path}")
        
        # Statistiche joint
        print(f"\nJoint error statistics:")
        for joint_name in sorted(unique_joints):
            df_joint_filtered = df_joint_all[df_joint_all["joint_name"] == joint_name]
            compare_types = df_joint_filtered['compare_type'].dropna().unique()
            compare_type = compare_types[0] if len(compare_types) > 0 else 'position'
            mae_values = df_joint_filtered["mae_pos"]
            unit = 'rad/s' if compare_type == 'velocity' else 'rad'
            print(f"  {joint_name}: ({compare_type})")
            print(f"    Mean: {mae_values.mean():.6f} {unit}")
            print(f"    Std:  {mae_values.std():.6f} {unit}")
            print(f"    Min:  {mae_values.min():.6f} {unit}")
            print(f"    Max:  {mae_values.max():.6f} {unit}")

        # --- Timeseries aggregation per joint (sovrapposto + media±std) ---
        for joint_name in sorted(unique_joints):
            # Collect timeseries from each version
            series_per_version = {}
            for v in range(1, 6):
                version = f"v{v}"
                ts_path = output_base_dir / version / "state_vs_cmd" / "compare_timeseries" / f"err_{safe_name(joint_name)}.csv"
                # also need t_start/t_end for normalization: read from aggregated df_joint_all
                df_match = df_joint_all[(df_joint_all["version"] == version) & (df_joint_all["joint_name"] == joint_name)]
                if not ts_path.exists() or df_match.empty:
                    continue
                try:
                    df_ts = pd.read_csv(ts_path)
                except Exception:
                    continue
                t_start = df_match["t_start"].values[0]
                t_end = df_match["t_end"].values[0]
                t = df_ts["t"].values
                err = df_ts["err"].values
                # normalize if possible
                if not (pd.isna(t_start) or pd.isna(t_end)) and (t_end - t_start) != 0:
                    t_norm = (t - t_start) / (t_end - t_start)
                    mask = (t_norm >= 0) & (t_norm <= 1)
                    series_per_version[v] = (t_norm[mask], err[mask])
                else:
                    series_per_version[v] = (t, err)

            if not series_per_version:
                continue

            # Overlapped timeseries
            plt.figure(figsize=(12, 6))
            for v, (t_s, err_s) in sorted(series_per_version.items()):
                col = colors.get(v, None)
                plt.plot(t_s, err_s, label=f"v{v}", color=col, alpha=0.8)
            plt.xlabel("Normalized time (0=start, 1=end)")
            # choose units for this joint
            compare_types = df_joint_all[df_joint_all['joint_name'] == joint_name]['compare_type'].dropna().unique()
            compare_type = compare_types[0] if len(compare_types) > 0 else 'position'
            ylabel = 'error [rad/s]' if compare_type == 'velocity' else 'error [rad]'
            ctrl_title = ' (velocity control)' if compare_type == 'velocity' else ' (position control)'
            plt.ylabel(ylabel)
            plt.title(f"Error over time - {joint_name}{ctrl_title}")
            plt.legend(loc='best')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            png_path = unificati_dir / f"timeseries_{safe_name(joint_name)}.png"
            plt.savefig(png_path, dpi=150)
            plt.close()
            print(f"Saved: {png_path}")

            # Mean ± std over normalized grid
            t_grid = np.linspace(0, 1, 500)
            interp_list = []
            for v, (t_s, err_s) in sorted(series_per_version.items()):
                try:
                    interp = np.interp(t_grid, t_s, err_s, left=np.nan, right=np.nan)
                    interp_list.append(interp)
                except Exception:
                    pass
            if not interp_list:
                continue
            arr = np.array(interp_list)
            mean_err = np.nanmean(arr, axis=0)
            std_err = np.nanstd(arr, axis=0)

            plt.figure(figsize=(12,6))
            plt.plot(t_grid, mean_err, color='black', linewidth=2, label='mean')
            plt.fill_between(t_grid, mean_err-std_err, mean_err+std_err, color='gray', alpha=0.3, label='±1 std')
            plt.xlabel("Normalized time (0=start, 1=end)")
            plt.ylabel(ylabel)
            plt.title(f"Mean error over time - {joint_name}{ctrl_title}")
            plt.legend(loc='best')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            png_path = unificati_dir / f"timeseries_mean_{safe_name(joint_name)}.png"
            plt.savefig(png_path, dpi=150)
            plt.close()
            print(f"Saved: {png_path}")
    
    # Stampa statistiche
    if vicon_errors:
        df_all = pd.concat(vicon_errors, ignore_index=True)
        print(f"\nVicon error statistics (MAE v_x):")
        print(f"  Mean: {df_all['mae_vx'].mean():.6f} m/s")
        print(f"  Std:  {df_all['mae_vx'].std():.6f} m/s")
        print(f"  Min:  {df_all['mae_vx'].min():.6f} m/s")
        print(f"  Max:  {df_all['mae_vx'].max():.6f} m/s")

def safe_name(label: str) -> str:
    """Convert a label into a filesystem-safe name"""
    return label.replace("/", "_").replace("[", "_").replace("]", "")

def main():
    ap = argparse.ArgumentParser(description="Parent script to run benchmarks")
    ap.add_argument("--test_name", required=True, help="Test name (e.g. 'avanti', 'discesa_5deg')")
    ap.add_argument("--versions", default="1,2,3,4,5", help="Versions to run (default: 1,2,3,4,5)")
    args = ap.parse_args()
    
    test_name = args.test_name
    requested_versions = [int(v) for v in args.versions.split(",")]
    
    # Definisci i path
    project_root = get_project_root()
    rosbag_base = get_rosbag_dir()
    test_rosbag_dir = rosbag_base / test_name
    output_base_dir = project_root / "output" / test_name
    
    # Check test exists
    if not test_rosbag_dir.exists():
        print(f"Error: test folder not found: {test_rosbag_dir}")
        sys.exit(1)
    
    # Trova tutte le directory di versione disponibili
    available_versions = find_version_directories(test_rosbag_dir)
    
    if not available_versions:
        print(f"Error: no version directories found in {test_rosbag_dir}")
        print(f"Looking for directories matching: benchmark_*_v1, benchmark_*_v2, etc.")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK FOR TEST: {test_name}")
    print(f"{'='*70}")
    print(f"Rosbag base dir: {test_rosbag_dir}")
    print(f"Output base dir: {output_base_dir}")
    print(f"Found version directories: {sorted(available_versions.keys())}")
    print(f"Versions to run: {requested_versions}")
    
    # Remove the output folder if it already exists
    if output_base_dir.exists():
        print(f"\n[INFO] Removing existing output folder: {output_base_dir}")
        shutil.rmtree(output_base_dir)
        print(f"[INFO] Removed. Creating new structure...")
    
    # Create the output directory
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # For each requested version, run the extraction scripts
    for version_num in requested_versions:
        if version_num not in available_versions:
            print(f"\nWarning: version v{version_num} not found")
            continue
        
        version_rosbag_dir = available_versions[version_num]
        
        # Find .mcap files in the version directory
        mcap_files = find_mcap_files(version_rosbag_dir)
        
        if not mcap_files:
            print(f"\nWarning: no .mcap files found in {version_rosbag_dir}")
            continue
        
        # Use the first .mcap file found (usually one per version)
        mcap_file = mcap_files[0]
        print(f"\n{'*'*70}")
        print(f"Processing v{version_num}: {version_rosbag_dir.name}")
        print(f"File: {mcap_file.name}")
        print(f"{'*'*70}")
        
        success = run_extraction_scripts(test_name, version_num, mcap_file, output_base_dir)
        
        if not success:
            print(f"\nError while processing {version_rosbag_dir.name}. Continuing with next versions...")
            continue
    
    # Aggrega gli errori
    aggregate_errors(test_name, output_base_dir)
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETED FOR TEST: {test_name}")
    print(f"Output saved in: {output_base_dir}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
