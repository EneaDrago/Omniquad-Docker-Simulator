#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script padre che esegue i benchmark per un test specifico.

Uso:
    python run_benchmark.py --test_name avanti
    python run_benchmark.py --test_name discesa_5deg

Struttura attesa:
  rosbag/REAL_ROBOT_rough_GIOVEDI/{test_name}/benchmark_*_v1/
  rosbag/REAL_ROBOT_rough_GIOVEDI/{test_name}/benchmark_*_v2/
  ...
  rosbag/REAL_ROBOT_rough_GIOVEDI/{test_name}/benchmark_*_v5/

Output salvato in:
  script_benchmark/output/{test_name}/
    v1/, v2/, ..., v5/ (ciascuno con vicon/, joint/)
    unificati/ (file aggregati di errori)
"""

import argparse
import os
import sys
import subprocess
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
    """Esegue i due script di estrazione per un file mcap"""
    
    script_dir = get_project_root()
    version_str = f"v{version_num}"
    
    # Script 1: extract_mcap_body_vs_wheels.py
    script1 = script_dir / "extract_mcap_body_vs_wheels.py"
    output_dir_vicon = output_dir / version_str / "vicon"
    output_dir_vicon.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Esecuzione: {script1.name}")
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
        print(f"Errore durante l'esecuzione di {script1.name}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    
    # Script 2: extract_joint_cmd_vs_state.py
    script2 = script_dir / "extract_joint_cmd_vs_state.py"
    output_dir_joint = output_dir / version_str / "joint"
    output_dir_joint.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Esecuzione: {script2.name}")
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
        print(f"Errore durante l'esecuzione di {script2.name}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    
    return True

def aggregate_errors(test_name, output_base_dir):
    """Aggrega gli errori dalle diverse versioni"""
    
    unificati_dir = output_base_dir / "unificati"
    unificati_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("Aggregazione errori dalle diverse versioni")
    print(f"{'='*70}")
    
    # Leggi i file velocity_error_summary.csv da tutte le versioni
    vicon_errors = []
    joint_errors = []
    
    for v in range(1, 6):
        version = f"v{v}"
        
        # Errori Vicon
        vicon_csv = output_base_dir / version / "vicon" / "velocity_error_summary.csv"
        if vicon_csv.exists():
            df = pd.read_csv(vicon_csv)
            df["version"] = version
            vicon_errors.append(df)
        
        # TODO: Aggiungi lettura degli errori joint quando disponibili
    
    # Aggrega errori Vicon
    if vicon_errors:
        df_vicon_all = pd.concat(vicon_errors, ignore_index=True)
        
        # Salva CSV aggregato
        csv_path = unificati_dir / "confronto_errori_vicon.csv"
        df_vicon_all.to_csv(csv_path, index=False)
        print(f"Salvato: {csv_path}")
        
        # Crea plot aggregato
        plt.figure(figsize=(10, 6))
        plt.bar(df_vicon_all["version"], df_vicon_all["mae_vx"])
        plt.xlabel("Versione")
        plt.ylabel("MAE v_x [m/s]")
        plt.title(f"Confronto errori Vicon - Test: {test_name}")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        png_path = unificati_dir / "confronto_errori_vicon.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Salvato: {png_path}")
    
    # Stampa statistiche
    if vicon_errors:
        df_all = pd.concat(vicon_errors, ignore_index=True)
        print(f"\nStatistiche errori Vicon (MAE v_x):")
        print(f"  Media: {df_all['mae_vx'].mean():.6f} m/s")
        print(f"  Std:   {df_all['mae_vx'].std():.6f} m/s")
        print(f"  Min:   {df_all['mae_vx'].min():.6f} m/s")
        print(f"  Max:   {df_all['mae_vx'].max():.6f} m/s")

def main():
    ap = argparse.ArgumentParser(description="Script padre per eseguire i benchmark")
    ap.add_argument("--test_name", required=True, help="Nome del test (es. 'avanti', 'discesa_5deg')")
    ap.add_argument("--versions", default="1,2,3,4,5", help="Versioni da eseguire (default: 1,2,3,4,5)")
    args = ap.parse_args()
    
    test_name = args.test_name
    requested_versions = [int(v) for v in args.versions.split(",")]
    
    # Definisci i path
    project_root = get_project_root()
    rosbag_base = get_rosbag_dir()
    test_rosbag_dir = rosbag_base / test_name
    output_base_dir = project_root / "output" / test_name
    
    # Verifica che il test esista
    if not test_rosbag_dir.exists():
        print(f"Errore: cartella test non trovata: {test_rosbag_dir}")
        sys.exit(1)
    
    # Trova tutte le directory di versione disponibili
    available_versions = find_version_directories(test_rosbag_dir)
    
    if not available_versions:
        print(f"Errore: nessuna directory di versione trovata in {test_rosbag_dir}")
        print(f"Cercate directory nel formato: benchmark_*_v1, benchmark_*_v2, etc.")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK PER TEST: {test_name}")
    print(f"{'='*70}")
    print(f"Rosbag base dir: {test_rosbag_dir}")
    print(f"Output base dir: {output_base_dir}")
    print(f"Directory versioni trovate: {sorted(available_versions.keys())}")
    print(f"Versioni da eseguire: {requested_versions}")
    
    # Crea la directory di output
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Per ogni versione richiesta, esegui gli script
    for version_num in requested_versions:
        if version_num not in available_versions:
            print(f"\nAvviso: versione v{version_num} non trovata")
            continue
        
        version_rosbag_dir = available_versions[version_num]
        
        # Trova file mcap nella directory della versione
        mcap_files = find_mcap_files(version_rosbag_dir)
        
        if not mcap_files:
            print(f"\nAvviso: nessun file .mcap trovato in {version_rosbag_dir}")
            continue
        
        # Usa il primo file mcap trovato (di solito c'è solo un file per versione)
        mcap_file = mcap_files[0]
        print(f"\n{'*'*70}")
        print(f"Elaborazione v{version_num}: {version_rosbag_dir.name}")
        print(f"File: {mcap_file.name}")
        print(f"{'*'*70}")
        
        success = run_extraction_scripts(test_name, version_num, mcap_file, output_base_dir)
        
        if not success:
            print(f"\nErrore durante l'elaborazione di {version}. Continuando con le versioni successive...")
            continue
    
    # Aggrega gli errori
    aggregate_errors(test_name, output_base_dir)
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETATO PER TEST: {test_name}")
    print(f"Output salvato in: {output_base_dir}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
