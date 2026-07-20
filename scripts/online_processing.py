#!/usr/bin/env python3
"""
RaTag Online DAQ Monitor.
Polls the raw data directory for new, stable acquisitions and dynamically 
triggers the main.py orchestrator.
"""
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

from RaTag.io.file_ops import load_yaml

# Import the orchestrator's main function directly
from main import main as execute_main_pipeline


def evaluate_directory_stability(current_count: int, last_count: int, current_ticks: int, threshold: int) -> tuple[int, int, bool]:
    """
    Evaluates the stability of a directory based on file counts.
    Returns: (updated_count, updated_ticks, is_ready_to_process)
    """
    if current_count == 0:
        return 0, 0, False

    if current_count == last_count:
        new_ticks = current_ticks + 1
        is_ready = new_ticks >= threshold
        return current_count, new_ticks, is_ready
    else:
        # File count changed; reset ticks
        return current_count, 0, False


def monitor_and_process(config_path: Path, poll_interval: int = 10, stable_threshold: int = 2, global_timeout_ticks: int = 30):
    """
    Monitors the data directory for new folders. Breaks gracefully if no new data 
    appears after global_timeout_ticks.
    """
    config = load_yaml(config_path)
    run_dir = Path(config['data']['raw_data_path'])
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Raw data path does not exist: {run_dir}")
        
    print(f"\n{'='*60}")
    print(f"STARTING ONLINE DAQ MONITOR: {config.get('run_id', 'UNKNOWN')}")
    print(f"Watching: {run_dir}")
    print(f"Polling Interval: {poll_interval}s | Stability Threshold: {stable_threshold} ticks")
    print(f"Global Timeout: {global_timeout_ticks * poll_interval}s of total inactivity")
    print(f"{'='*60}\n")
    
    processed_dirs = set()
    file_counts = {}
    stable_ticks = {}
    idle_ticks = 0
    
    try:
        while True:
            run_is_complete = (run_dir / "RUN_COMPLETE.lock").exists()

            current_dirs = [d for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            unprocessed_dirs = [d for d in current_dirs if d not in processed_dirs]
            
            # 2. Timeout & Graceful Exit Logic
            if not unprocessed_dirs:
                if run_is_complete:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] RUN_COMPLETE.lock detected and all directories processed. Exiting gracefully.")
                    break
                    
                idle_ticks += 1
                if idle_ticks >= global_timeout_ticks:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Global timeout reached. Assuming run is complete. Exiting.")
                    break
            else:
                idle_ticks = 0
                
            current_dirs = [d for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            unprocessed_dirs = [d for d in current_dirs if d not in processed_dirs]
            
            # Global Timeout Logic
            if not unprocessed_dirs:
                idle_ticks += 1
                if idle_ticks >= global_timeout_ticks:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] No new directories for {global_timeout_ticks * poll_interval} seconds. Assuming run is complete. Exiting.")
                    break
            else:
                idle_ticks = 0
            
            triggered = False
            
            for d in unprocessed_dirs:
                current_count = len(list(d.glob('*.wfm')))
                last_count = file_counts.get(d, -1)
                current_tick = stable_ticks.get(d, 0)
                
                # Delegate to the pure helper function
                new_count, new_tick, is_stable = evaluate_directory_stability(current_count, last_count, current_tick, stable_threshold)
                
                file_counts[d] = new_count
                stable_ticks[d] = new_tick
                
                if is_stable:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] >>> ACQUISITION COMPLETE: {d.name} ({new_count} files)")
                    triggered = True
                    processed_dirs.add(d)
            
            # Execute Main Pipeline
            if triggered:
                print(f"Delegating execution to main.py orchestrator...")
                
                # Temporarily patch sys.argv so main() can parse the config path natively
                original_argv = sys.argv.copy()
                sys.argv = ['main.py', str(config_path)]
                
                try:
                    execute_main_pipeline()
                finally:
                    # Restore original arguments to prevent state bleed
                    sys.argv = original_argv
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing complete. Resuming monitoring...\n")
            
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        print("\nOnline monitoring terminated by user.")


def main():
    parser = argparse.ArgumentParser(description='RaTag Online DAQ Monitor')
    parser.add_argument('config', type=Path, help='Path to YAML config file')
    parser.add_argument('--poll', type=int, default=10, help='Polling interval in seconds (default: 10)')
    parser.add_argument('--ticks', type=int, default=2, help='Consecutive stable ticks required (default: 2)')
    parser.add_argument('--timeout', type=int, default=30, help='Idle ticks before graceful exit (default: 30)')
    args = parser.parse_args()

    monitor_and_process(args.config, poll_interval=args.poll, stable_threshold=args.ticks, global_timeout_ticks=args.timeout)

if __name__ == "__main__":
    main()