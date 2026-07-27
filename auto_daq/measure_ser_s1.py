import os
import pyvisa
import argparse

# ---------------------------------------------------------
# EXPLICIT PHYSICS CONFIGURATIONS
# ---------------------------------------------------------
CONFIGS = {
    "SER": {
        "v_scale": 10e-3,       # 10 mV/div
        "h_scale": 200e-9,      # 200 ns/div
        "frames_per_batch": 1186,
        "default_batches": 1000
    },
    "S1": {
        "v_scale": 40e-3,       # 40 mV/div
        "h_scale": 2e-6,        # 2 us/div
        "frames_per_batch": 124,
        "default_batches": 2500
    }
}

def run_measurement(
    measurement_type: str,
    measurement_index: int,
    num_batches: int = None,
    scope_address: str = 'TCPIP0::132.72.12.214::inst0::INSTR',
    channel: str = 'CH1'
):
    """Explicit, data-driven DAQ pipeline."""
    
    if measurement_type not in CONFIGS:
        raise ValueError(f"Measurement type '{measurement_type}' is not defined in CONFIGS.")

    # 1. Load parameters dynamically
    cfg = CONFIGS[measurement_type]
    batches = num_batches if num_batches is not None else cfg["default_batches"]
    frames = cfg["frames_per_batch"]
    
    # 2. Path generation
    save_directory = rf"L:\RUN2\P{measurement_index}"
    os.makedirs(save_directory, exist_ok=True)
    file_prefix = f"RUN2_150726-Cath600_Gate00_Anode00_25bar_175T_{measurement_type}_P{measurement_index}_"

    rm = pyvisa.ResourceManager()
    scope = rm.open_resource(scope_address)
    scope.timeout = 60000 

    try:
        scope.write('*CLS')
        print(f"Connected to: {scope.query('*IDN?').strip()}")
        scope.write('ACQUIRE:STATE STOP')

        # 3. EXPLICIT PHYSICS STATE: Apply loaded parameters
        scope.write(f'{channel}:SCAle {cfg["v_scale"]}')
        scope.write(f'HORizontal:SCAle {cfg["h_scale"]}')
        scope.write('HORizontal:MODE:SAMPLERate 5e9')

        scope.write('HORizontal:FASTframe:STATE ON')
        scope.write(f'HORizontal:FASTframe:COUNt {frames}')
        scope.write('ACQUIRE:STOPAFTER SEQUENCE')
        scope.write('SAVE:WAVEFORM:FILEFORMAT INTERNAL')

        esr = int(scope.query('*ESR?').strip())
        if esr != 0:
            print(f"Setup Error: ESR={esr} | {scope.query('ALLEV?').strip()}")
            return

        print(f"[{measurement_type}] Starting {batches} batches of {frames} frames...")
        print(f"Saving to: {save_directory}")
        print("--------------------------------------------------")

        # 4. The Flat DAQ Loop
        for batch_idx in range(batches):
            print(f"Batch {batch_idx + 1}/{batches}: Arming trigger...", end="\r")
            
            scope.write('ACQUIRE:STATE RUN')
            scope.query('*OPC?') # Block for physics

            filename = f"{save_directory}\\{file_prefix}{batch_idx + 1:04d}.wfm"
            scope.write(f'SAVE:WAVEFORM {channel}, "{filename}"')
            
            scope.query('*OPC?') # Block for disk I/O

            esr = int(scope.query('*ESR?').strip())
            if esr != 0:
                print(f"\nBatch {batch_idx + 1} Error during save: ESR={esr} | {scope.query('ALLEV?').strip()}")
                break
                
        print(f"\n[{measurement_type}] Measurement set P{measurement_index} complete.")

    except KeyboardInterrupt:
        print("\nDAQ aborted manually via Ctrl+C.")
        raise
    except Exception as e:
        print(f"\nDAQ pipeline crashed: {e}")
        raise
    finally:
        try:
            scope.write('ACQUIRE:STATE STOP')
            scope.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data-Driven Oscilloscope DAQ")
    
    # Automatically pull choices from the CONFIGS dictionary
    parser.add_argument("type", type=str, choices=list(CONFIGS.keys()), help="Measurement type from CONFIGS")
    parser.add_argument("index", type=int, help="Measurement index (n) for the P{n} directory")
    parser.add_argument("--batches", type=int, default=None, help="Override default number of batches")
    
    args = parser.parse_args()
    
    run_measurement(
        measurement_type=args.type, 
        measurement_index=args.index, 
        num_batches=args.batches
    )