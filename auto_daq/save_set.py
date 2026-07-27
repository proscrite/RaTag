import pyvisa

def run_fastframe_daq_pipeline(
    file_prefix: str,
    save_directory: str = r'P:\AXO-RITA\raw_data\RUN34_EL25k',
    scope_address: str = 'TCPIP0::132.72.13.144::inst0::INSTR',
    num_batches: int = 50,
    frames_per_batch: int = 988,
    active_channels: list = ['Ch3', 'Ch4'],
    trigger_source: str = 'Ch4'
):
    """
    Explicitly drives a FastFrame DAQ loop from Python for a single set of parameters.
    Blocks on physics acquisition and blocks on disk I/O.
    Saves multiple active channels simultaneously.
    """
    rm = pyvisa.ResourceManager()
    scope = rm.open_resource(scope_address)
    
    # CRITICAL: Raise the timeout to 1.5 hours (5,400,000 ms). 
    # Python will hang here gracefully while waiting for recoil triggers.
    scope.timeout = 60000

    try:
        # 1. Initialize and clear errors
        scope.write('*CLS')
        idn = scope.query('*IDN?').strip()
        print(f"Connected to: {idn}")

        # 2. Stop acquisition to safely configure hardware
        scope.write('ACQUIRE:STATE STOP')

        # 3. EXPLICIT PHYSICS STATE: Channel & Trigger Management
        # (Commented out: firmware treats ON as a toggle rather than absolute state)
        # for ch in [1, 2, 3, 4]:
        #     ch_str = f'Ch{ch}'
        #     state = 'ON' if ch_str in active_channels else 'OFF'
        #     scope.write(f'SELect:{ch_str} {state}')

        # Lock the trigger source explicitly
        scope.write(f'TRIGger:A:EDGE:SOUrce {trigger_source}')

        # 4. Explicit FastFrame Configuration
        scope.write('HORizontal:FASTframe:STATE ON')
        scope.write(f'HORizontal:FASTframe:COUNt {frames_per_batch}')

        # 5. Explicit trigger and file format setup
        scope.write('ACQUIRE:STOPAFTER SEQUENCE')
        scope.write('SAVE:WAVEFORM:FILEFORMAT INTERNAL')

        # Flat inline error check for setup
        esr = int(scope.query('*ESR?').strip())
        if esr != 0:
            allev = scope.query('ALLEV?').strip()
            print(f"Setup Error: ESR={esr} | {allev}")
            return

        print(f"Active Channels locked to: {active_channels}")
        print(f"Trigger locked to: {trigger_source}")
        print(f"Starting {num_batches} batches of {frames_per_batch} frames...")
        print("--------------------------------------------------")

        # 6. The Flat DAQ Loop
        for batch_idx in range(num_batches):
            print(f"Batch {batch_idx + 1}/{num_batches}: Arming trigger...")
            
            # A. Start the acquisition sequence
            scope.write('ACQUIRE:STATE RUN')
            
            # B. Wait for Physics (Blocks until frames are full)
            scope.query('*OPC?')
            print(f"Batch {batch_idx + 1}: Frame stack full. Saving to disk...")

            # C. Explicit Save Command for ALL active channels
            # The scope automatically appends "_CH3.wfm" and "_CH4.wfm" to the base name.
            # E.g. P:\...\RUN34_20260706_Gate0200_Anode2100_001
            filename_base = f"{save_directory}\\{file_prefix}_Batch{batch_idx + 1:03d}_"
            scope.write(f'SAVE:WAVEFORM ALL, "{filename_base}"')
            
            # D. Wait for Disk I/O (Blocks until file write is fully complete)
            scope.query('*OPC?')

            # E. Inline explicit error check
            esr = int(scope.query('*ESR?').strip())
            if esr != 0:
                allev = scope.query('ALLEV?').strip()
                print(f"Batch {batch_idx + 1} Error during save: ESR={esr} | {allev}")
                break
            else:
                print(f"Batch {batch_idx + 1} successfully saved with prefix {filename_base}")

    except KeyboardInterrupt:
        print("\nDAQ aborted manually via Ctrl+C.")
    except Exception as e:
        print(f"\nDAQ pipeline crashed: {e}")
    finally:
        # Guarantee the instrument isn't left hung
        try:
            scope.write('ACQUIRE:STATE STOP')
            scope.close()
            print("Instrument connection closed safely.")
        except Exception:
            pass

if __name__ == "__main__":
    # Example execution matching the explicit parameters requested
    run_fastframe_daq_pipeline(
        file_prefix="RUN34_20260706_Gate0200_Anode2100",
        save_directory=r"P:\AXO-RITA\raw_data\RUN34_EL25k",
        num_batches=2  # Set to 2 for quick testing, change to 50 for the real run
    )