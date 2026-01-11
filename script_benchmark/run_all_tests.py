#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to automatically discover and run benchmarks for all tests in rosbag/REAL_ROBOT_rough_GIOVEDI.

Usage:
    python run_all_tests.py
    python run_all_tests.py --tests avanti discesa_5deg  # Run specific tests only
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

def get_project_root():
    """Return the path of the script_benchmark folder"""
    return Path(__file__).parent

def get_rosbag_dir():
    """Return the path of rosbag/REAL_ROBOT_rough_GIOVEDI"""
    project_root = get_project_root()
    return project_root.parent / "rosbag" / "REAL_ROBOT_rough_GIOVEDI"

def discover_tests(rosbag_dir):
    """
    Discover all test directories in rosbag_dir.
    Returns a sorted list of test names.
    """
    rosbag_path = Path(rosbag_dir)
    if not rosbag_path.exists():
        return []
    
    tests = []
    for item in sorted(rosbag_path.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            tests.append(item.name)
    
    return tests

def run_benchmark(test_name):
    """Run benchmark for a single test"""
    script_dir = get_project_root()
    cmd = [sys.executable, str(script_dir / "run_benchmark.py"), "--test_name", test_name]
    
    print(f"\n{'='*80}")
    print(f"Running benchmark for: {test_name}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running benchmark for {test_name}:")
        print(f"  Exit code: {e.returncode}")
        return False

def main():
    ap = argparse.ArgumentParser(
        description="Automatically discover and run all benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_tests.py                    # Run all tests
  python run_all_tests.py --tests avanti     # Run specific test(s)
        """
    )
    ap.add_argument("--tests", nargs='+', default=None, 
                    help="Specific test names to run (default: all)")
    ap.add_argument("--skip", nargs='+', default=[], 
                    help="Test names to skip")
    args = ap.parse_args()
    
    # Discover available tests
    rosbag_dir = get_rosbag_dir()
    available_tests = discover_tests(rosbag_dir)
    
    if not available_tests:
        print(f"Error: No test directories found in {rosbag_dir}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK SUITE - AUTO DISCOVERY")
    print(f"{'='*80}")
    print(f"Rosbag directory: {rosbag_dir}")
    print(f"Available tests: {', '.join(available_tests)}")
    print(f"{'='*80}\n")
    
    # Determine which tests to run
    if args.tests:
        tests_to_run = args.tests
        # Verify all requested tests exist
        invalid = [t for t in tests_to_run if t not in available_tests]
        if invalid:
            print(f"Error: Invalid test names: {invalid}")
            print(f"Available tests: {available_tests}")
            sys.exit(1)
    else:
        tests_to_run = available_tests
    
    # Remove skipped tests
    tests_to_run = [t for t in tests_to_run if t not in args.skip]
    
    if not tests_to_run:
        print("No tests to run.")
        sys.exit(0)
    
    print(f"Tests to run ({len(tests_to_run)}): {', '.join(tests_to_run)}")
    print(f"{'='*80}\n")
    
    # Run benchmarks
    start_time = datetime.now()
    results = {}
    
    for i, test_name in enumerate(tests_to_run, 1):
        print(f"\n[{i}/{len(tests_to_run)}] Starting: {test_name}")
        success = run_benchmark(test_name)
        results[test_name] = "✓ PASSED" if success else "✗ FAILED"
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    # Print summary
    print(f"\n\n{'='*80}")
    print(f"BENCHMARK SUITE SUMMARY")
    print(f"{'='*80}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:   {elapsed}")
    print(f"{'='*80}\n")
    
    for test_name, result in results.items():
        print(f"  {test_name:30s} {result}")
    
    print(f"\n{'='*80}")
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    print(f"{'='*80}\n")
    
    # Exit with error if any test failed
    if passed < total:
        sys.exit(1)

if __name__ == "__main__":
    main()
