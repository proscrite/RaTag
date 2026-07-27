import pyvisa
import time
import csv
import argparse
import sys
from datetime import datetime

# --- HARDWARE CONFIGURATION ---
COM_PORT = "COM3"  
BAUD_RATE = 9600   

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

def main():
    parser = argparse.ArgumentParser(description="CAEN HV Continuous Monitor")
    parser.add_argument("log_file", type=str, help="Absolute path to the CSV log file")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    args = parser.parse_args()

    rm = pyvisa.ResourceManager()
    
    try:
        inst = rm.open_resource(COM_PORT)
        inst.baud_rate = BAUD_RATE
        inst.data_bits = 8
        inst.write_termination = '\r\n'
        inst.read_termination = '\r\n'
        inst.timeout = 2000
        
        # We don't print heavily here because stdout might be mixed with the master script.
        # But we do log silently in the background.
        while True:
            v0_now = get_vmon(inst, 0)
            v1_now = get_vmon(inst, 1)
            v2_now = get_vmon(inst, 2)
            v3_now = get_vmon(inst, 3)
            i0_now = get_imon(inst, 0)
            i1_now = get_imon(inst, 1)
            i2_now = get_imon(inst, 2)
            i3_now = get_imon(inst, 3)
            log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # File should exist (created by ramp_and_log_hv), but we safety-check just in case.
            try:
                with open(args.log_file, 'x', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Timestamp", "CH0_V", "CH1_V", "CH2_V", "CH3_V", "CH0_I", "CH1_I", "CH2_I", "CH3_I"])
            except FileExistsError:
                pass 

            # Append the continuous data
            with open(args.log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([log_time, v0_now, v1_now, v2_now, v3_now, i0_now, i1_now, i2_now, i3_now])
            
            # Sleep until the next poll. 
            # If terminated by the master script, this sleep gets interrupted gracefully.
            time.sleep(args.interval)

    except KeyboardInterrupt:
        # Expected if the parent passes down a Ctrl+C
        pass
    except Exception as e:
        print(f"Background Monitor Failed: {e}", file=sys.stderr)
    finally:
        if 'inst' in locals():
            inst.close()
        rm.close()

if __name__ == "__main__":
    main()