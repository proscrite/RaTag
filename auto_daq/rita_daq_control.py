import os
import time
import datetime
import subprocess
import save_set
import caen_ramp

def run_master_field_scan():
    """
    Master pipeline to orchestrate CAEN high-voltage ramping, 
    continuous background voltage monitoring, and Tektronix data acquisition.
    """
    # ---------------------------------------------------------
    # EXPLICIT PHYSICS & PATH PARAMETERS
    # ---------------------------------------------------------
    run_id = 34
    base_path = r"P:\AXO-RITA\raw_data\RUN34_EL25k"
    
    gate_voltages = [768, 672, 576, 280, 384, 288, 252, 216, 188, 120]
    voltage_difference = 1000  # Anode is Gate + 1000V
    
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    print(f"Starting Master Field Scan for RUN {run_id}")
    print(f"Base Path: {base_path}")
    print("--------------------------------------------------")

    try:
        # ---------------------------------------------------------
        # THE FLAT DAQ ORCHESTRATION LOOP
        # ---------------------------------------------------------
        for gate in gate_voltages:
            anode = gate + voltage_difference
            
            folder_name = f"FieldScan_Gate{gate:04d}_Anode{anode:04d}"
            target_directory = os.path.join(base_path, folder_name)
            file_prefix = f"RUN{run_id}_{current_date}_Gate{gate:04d}_Anode{anode:04d}_"
            hv_log_path = os.path.join(target_directory, f"hv_log_{current_date}.csv")
            
            os.makedirs(target_directory, exist_ok=True)
            print(f"\n[{folder_name}] Setup Target Directory: {target_directory}")

            # 1. RAMP: Command the CAEN Power Supply (Imported module)
            print(f"[{folder_name}] Ramping CAEN: CH1 (Gate) -> {gate}V, CH2 (Anode) -> {anode}V")
            caen_ramp.ramp_and_log_hv(
                ch1_target=gate, 
                ch2_target=anode, 
                log_file=hv_log_path
            )
                
            print(f"[{folder_name}] Ramping complete. Waiting 5 seconds for stabilization...")
            time.sleep(5)
            
            # 2. CONTINUOUS MONITOR: Spawn the background process
            print(f"[{folder_name}] Starting background CAEN monitoring (60s interval)...")
            monitor_process = subprocess.Popen([
                "python", "caen_monitor.py", 
                hv_log_path, 
                "--interval", "60"
            ])
            
            try:
                # 3. PHYSICS DAQ: Execute Oscilloscope Data Acquisition
                print(f"[{folder_name}] Launching Tektronix FastFrame DAQ...")
                save_set.run_fastframe_daq_pipeline(
                    file_prefix=file_prefix,
                    save_directory=target_directory
                )
                print(f"[{folder_name}] Acquisition set complete.")
            
            finally:
                # 4. EXPLICIT CLEANUP: Always kill the monitor when DAQ ends
                print(f"[{folder_name}] Shutting down background monitor...")
                monitor_process.terminate()
                monitor_process.wait() # Ensure the OS fully releases the COM port
            
    except KeyboardInterrupt:
        print("\n\n[!!!] MASTER PIPELINE ABORTED BY USER [!!!]")
        print("Halting all future scans.")
    except Exception as e:
        print(f"\n\n[!!!] MASTER PIPELINE FAILED: {e}")
        
    print("\n--------------------------------------------------")
    print("Master Field Scan complete/terminated.")

if __name__ == "__main__":
    run_master_field_scan()