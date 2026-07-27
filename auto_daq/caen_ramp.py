import pyvisa
import time
import argparse
import csv
from datetime import datetime

# --- HARDWARE CONFIGURATION ---
COM_PORT = "COM3"  
BAUD_RATE = 9600   

# --- RAMP RATE PROTECTION ---
RUP_RATE = 20.0   # Fast ramp up is fine (20 V/s)
RDW_RATE = 50.0   # Faster ramp down (50 V/s) to prevent capacitive current trips
MAX_I = 15.0      # uA

# --- DELTA GUARDRAIL ---
MAX_DELTA_V = 630.0  
MICRO_STEP_SIZE = 20.0  # Volts per synchronized step
CURRENT_TRIP_LIMIT = 0.1 # Software kill-switch limit (uA)

def send_command(inst, cmd):
    """Sends an ASCII command and returns the stripped response."""
    return inst.query(cmd).strip()

def get_vmon(inst, channel):
    """Queries the Voltage Monitor for a specific channel."""
    resp = send_command(inst, f"$BD:00,CMD:MON,CH:{channel},PAR:VMON")
    if "VAL:" in resp:
        return float(resp.split("VAL:")[-1])
    return -1.0 

def get_imon(inst, channel):
    """Queries the Current Monitor for a specific channel."""
    resp = send_command(inst, f"$BD:00,CMD:MON,CH:{channel},PAR:IMON")
    if "VAL:" in resp:
        return float(resp.split("VAL:")[-1])
    return -1.0

def verify_static_channels(inst, ch0_v_target=850.0, ch0_i_limit=70.0, ch3_v_target=50.0):
    """
    Helper function to verify the static voltages/currents of the PMT (CH0) 
    and PIN detector (CH3). If they are currently at 0V, it automatically 
    ramps them to their targets before performing the final safety verification.
    """
    print("\n--- VERIFYING STATIC CHANNELS ---")
    v0_initial = get_vmon(inst, 0)
    v3_initial = get_vmon(inst, 3)
    
    needs_ramp = False

    # 1. Auto-Ramp PMT if starting from zero
    if v0_initial < 5.0:
        print(f"[*] CH0 (PMT) is at {v0_initial}V. Initiating auto-ramp to {ch0_v_target}V.")
        send_command(inst, f"$BD:00,CMD:SET,CH:0,PAR:ISET,VAL:{ch0_i_limit}")
        send_command(inst, f"$BD:00,CMD:SET,CH:0,PAR:RUP,VAL:20.0")  # Safe 20 V/s for PMT
        send_command(inst, f"$BD:00,CMD:SET,CH:0,PAR:RDW,VAL:20.0")
        send_command(inst, f"$BD:00,CMD:SET,CH:0,PAR:ON")
        send_command(inst, f"$BD:00,CMD:SET,CH:0,PAR:VSET,VAL:{ch0_v_target}")
        needs_ramp = True

    # 2. Auto-Ramp PIN if starting from zero
    if v3_initial < 5.0:
        print(f"[*] CH3 (PIN) is at {v3_initial}V. Initiating auto-ramp to {ch3_v_target}V.")
        send_command(inst, f"$BD:00,CMD:SET,CH:3,PAR:ISET,VAL:15.0")
        send_command(inst, f"$BD:00,CMD:SET,CH:3,PAR:RUP,VAL:10.0")  # Slower for PIN
        send_command(inst, f"$BD:00,CMD:SET,CH:3,PAR:RDW,VAL:10.0")
        send_command(inst, f"$BD:00,CMD:SET,CH:3,PAR:ON")
        send_command(inst, f"$BD:00,CMD:SET,CH:3,PAR:VSET,VAL:{ch3_v_target}")
        needs_ramp = True

    # 3. Wait for hardware to finish the auto-ramp
    if needs_ramp:
        print("\nWaiting for static channels to reach targets...")
        while True:
            v0_now = get_vmon(inst, 0)
            v3_now = get_vmon(inst, 3)
            print(f"Ramping -> CH0: {v0_now:06.1f}V | CH3: {v3_now:06.1f}V", end="\r")
            
            # Only wait for the channels we explicitly commanded
            ch0_ok = abs(v0_now - ch0_v_target) <= 2.0 if (v0_initial < 5.0) else True
            ch3_ok = abs(v3_now - ch3_v_target) <= 2.0 if (v3_initial < 5.0) else True
            
            if ch0_ok and ch3_ok:
                print("\n[*] Auto-ramp complete. Stabilizing...")
                time.sleep(5)
                break
            time.sleep(2.0)

    # 4. Strict Safety Verification (Final Check)
    v0_final = get_vmon(inst, 0)
    i0_final = get_imon(inst, 0)
    v3_final = get_vmon(inst, 3)

    print(f"\nStatic Check -> CH0 (PMT): {v0_final:06.1f}V, {i0_final:05.2f}uA | CH3 (PIN): {v3_final:06.1f}V")

    # Evaluate safety conditions (allowing a 2V tolerance)
    errors = []
    if abs(v0_final - ch0_v_target) > 2.0:
        errors.append(f"CH0 Voltage is {v0_final}V (Expected ~{ch0_v_target}V)")
    if i0_final > ch0_i_limit:
        errors.append(f"CH0 Current is {i0_final}uA (Exceeds {ch0_i_limit}uA limit)")
    if abs(v3_final - ch3_v_target) > 2.0:
        errors.append(f"CH3 Voltage is {v3_final}V (Expected ~{ch3_v_target}V)")

    if errors:
        emergency_shutdown(inst, "Static verification failed: " + " | ".join(errors))

    print("[SUCCESS] Static channels verified and stable.")
    
def emergency_shutdown(inst, reason):
    """Instantly kills power to protect hardware."""
    print(f"\n[!!!] EMERGENCY SHUTDOWN: {reason} [!!!]")
    send_command(inst, "$BD:00,CMD:SET,CH:1,PAR:OFF")
    send_command(inst, "$BD:00,CMD:SET,CH:2,PAR:OFF")
    print("Monitoring hardware discharge to safe levels (< 5V)...")
    
    # Active monitoring loop
    while True:
        v1_now = get_vmon(inst, 1)
        v2_now = get_vmon(inst, 2)
        i1_now = get_imon(inst, 1)
        i2_now = get_imon(inst, 2)
        
        # Print the decay live
        print(f"DISCHARGING -> VMon: CH1={v1_now:06.1f}V | CH2={v2_now:06.1f}V || IMon: CH1={i1_now:05.2f}uA | CH2={i2_now:05.2f}uA          ", end="\r")
        
        # Safe handling threshold
        if v1_now < 5.0 and v2_now < 5.0:
            break
            
        time.sleep(1)
        
    print("\n\n[SAFE] Hardware fully discharged. Closing connection.")
    inst.close()
    
    # Raise the error to ensure the master DAQ script knows a fault occurred
    raise RuntimeError(reason)

def calculate_next_voltage(current, target, step_size):
    """Calculates the next incremental voltage milestone towards a target."""
    if abs(current - target) <= step_size:
        return target
    elif current < target:
        return current + step_size
    else:
        return current - step_size

def ramp_and_log_hv(ch1_target: float, ch2_target: float, log_file: str):
    """
    Connects to the CAEN supply, synchronizes a co-ramp to the targets, 
    and logs the final voltages to the specified filepath.
    """
    # Pre-flight absolute check
    if abs(ch1_target - ch2_target) > MAX_DELTA_V:
        raise ValueError(f"ABORT: Final targets exceed the safe delta limit of {MAX_DELTA_V}V.")

    rm = pyvisa.ResourceManager()
    
    try:
        print(f"Connecting to CAEN on {COM_PORT}...")
        inst = rm.open_resource(COM_PORT)
        inst.baud_rate = BAUD_RATE
        inst.data_bits = 8
        inst.write_termination = '\r\n'
        inst.read_termination = '\r\n'
        inst.timeout = 2000
        
        # Configure configuration safety baselines
        for ch in [1, 2]:
            send_command(inst, f"$BD:00,CMD:SET,CH:{ch},PAR:ISET,VAL:{MAX_I}")
            send_command(inst, f"$BD:00,CMD:SET,CH:{ch},PAR:RUP,VAL:{RUP_RATE}")
            send_command(inst, f"$BD:00,CMD:SET,CH:{ch},PAR:RDW,VAL:{RDW_RATE}")
            send_command(inst, f"$BD:00,CMD:SET,CH:{ch},PAR:ON") # Ensure channels are enabled

        print("\n--- INITIATING SYNCHRONIZED CO-RAMP ---")
        
        while True:
            # Read current positions
            v1_now = get_vmon(inst, 1)
            v2_now = get_vmon(inst, 2)
            i1_now = get_imon(inst, 1)
            i2_now = get_imon(inst, 2)
            
            # Formatted string pad ensures the line overwrites cleanly
            print(f"Current VMon -> CH1: {v1_now:06.1f}V | CH2: {v2_now:06.1f}V ||      IMon -> CH1: {i1_now:05.2f}uA | CH2: {i2_now:05.2f}uA       ", end="\r")

           # 1. Dynamic Current Safety Check (Pre-breakdown leakage)
            if i1_now > CURRENT_TRIP_LIMIT or i2_now > CURRENT_TRIP_LIMIT:
                emergency_shutdown(inst, f"Overcurrent limit breached: CH1={i1_now}uA, CH2={i2_now}uA")

            # 2. Dynamic Voltage Safety Check
            if abs(v1_now - v2_now) > MAX_DELTA_V:
                emergency_shutdown(inst, f"Live delta violation: CH1={v1_now}V, CH2={v2_now}V")
            
            # Check if both targets are satisfied
            if abs(v1_now - ch1_target) <= 2.0 and abs(v2_now - ch2_target) <= 2.0:
                break

            # Calculate the next micro-step for both channels
            next_v1 = calculate_next_voltage(v1_now, ch1_target, MICRO_STEP_SIZE)
            next_v2 = calculate_next_voltage(v2_now, ch2_target, MICRO_STEP_SIZE)

            # Pre-check micro-steps before writing to hardware
            if abs(next_v1 - next_v2) > MAX_DELTA_V:
                emergency_shutdown(inst, f"Calculated micro-step unsafe: CH1={next_v1}V, CH2={next_v2}V")

            # Write micro-steps quasi-simultaneously
            send_command(inst, f"$BD:00,CMD:SET,CH:1,PAR:VSET,VAL:{next_v1}")
            send_command(inst, f"$BD:00,CMD:SET,CH:2,PAR:VSET,VAL:{next_v2}")

            # Brief pause to let the hardware processing catch up to the step
            time.sleep(1.5)

        print("\n\n[SUCCESS] Both channels stabilized at final target destinations.")

        # --- SINGLE-PASS FILE LOGGING ---
        v0_final = get_vmon(inst, 0)
        v1_final = get_vmon(inst, 1)
        v2_final = get_vmon(inst, 2)
        v3_final = get_vmon(inst, 3)

        i0_final = get_imon(inst, 0)
        i1_final = get_imon(inst, 1)
        i2_final = get_imon(inst, 2)
        i3_final = get_imon(inst, 3)
        log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            with open(log_file, 'x', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "CH0_V", "CH1_V", "CH2_V", "CH3_V", "CH0_I", "CH1_I", "CH2_I", "CH3_I"])
        except FileExistsError:
            pass 

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([log_time, v0_final, v1_final, v2_final, v3_final, i0_final, i1_final, i2_final, i3_final])
            
        print(f"Logged data to: {log_file}")

        # Check PMT and Si PIN static channels
        verify_static_channels(inst)

    finally:
        if 'inst' in locals():
            inst.close()
        rm.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAEN HV Ramping Script")
    parser.add_argument("--ch1", type=float, required=True, help="Target voltage for channel 1")
    parser.add_argument("--ch2", type=float, required=True, help="Target voltage for channel 2")
    parser.add_argument("--log_file", type=str, required=False, help="Absolute path to the CSV log file")
    args = parser.parse_args()

    ch1_target = args.ch1
    ch2_target = args.ch2

    log_file = args.log_file if args.log_file else f"hv_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    ramp_and_log_hv(ch1_target, ch2_target, log_file)