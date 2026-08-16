import os
import shutil
import subprocess
import numpy as np
import pandas as pd

def parse_final_positions(exyz_filepath):
    """
    Parses the SRIM EXYZ.txt file and returns a DataFrame of the 
    final stopping coordinates (X, Y, Z) for each ion.
    """
    skip_lines = 0
    with open(exyz_filepath, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('-------'):
                skip_lines = i + 1
                break
                
    col_names = ['Ion', 'Energy_keV', 'X_Ang', 'Y_Ang', 'Z_Ang', 'Elec_Stop', 'Recoil_Energy']
    df = pd.read_csv(exyz_filepath, delim_whitespace=True, skiprows=skip_lines, header=None, names=col_names)
    
    final_positions = df.groupby('Ion').last().reset_index()
    return final_positions[['X_Ang', 'Y_Ang', 'Z_Ang']]

def parse_escaped_ions(backscat_filepath):
    """
    Parses SRIM's BACKSCAT.txt to extract the residual kinetic energy 
    and exit angles of isotopes escaping into the gas phase.
    """
    # 1. Skip the header block
    skip_lines = 0
    with open(backscat_filepath, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('-------'):
                skip_lines = i + 1
                break
                
    # 2. Define the columns based on BACKSCAT.txt format
    col_names = ['Ion', 'Atom', 'Energy_eV', 'X', 'Y', 'Z', 'CosX', 'CosY', 'CosZ']
    
    # 3. Load the data
    df = pd.read_csv(backscat_filepath, delim_whitespace=True, skiprows=skip_lines, header=None, names=col_names)
    
    # 4. Clean up the data
    # We want the Energy in keV for consistency, and the exit angles
    escaped_data = pd.DataFrame()
    escaped_data['Ion'] = df['Ion']
    escaped_data['Exit_Energy_keV'] = df['Energy_eV'] / 1000.0
    
    # SRIM CosX is the depth axis. Since they are exiting the front face (X=0),
    # CosX will be negative. We can compute the angle relative to the surface normal.
    escaped_data['CosX'] = df['CosX']
    escaped_data['CosY'] = df['CosY']
    escaped_data['CosZ'] = df['CosZ']
    
    # Calculate exit angle (theta) relative to the surface normal (in degrees)
    escaped_data['Exit_Angle_Deg'] = np.degrees(np.arccos(np.abs(df['CosX'])))
    
    return escaped_data

def generate_trim_dat(output_path, num_ions, ion_z, ion_mass, energy_ev, coords_df=None, mode='forward'):
    """
    Generates TRIM.DAT for isotropic emission.
    mode='forward': CosX > 0 (for flat source at interface)
    mode='full': CosX between -1 and 1 (for implanted isotopes)
    """
    with open(output_path, 'w') as f:
        f.write(f"Isotropic emission: {mode}\n")
        f.write(f"{num_ions}\n")
        
        # Calculate random trajectory vectors
        if mode == 'forward':
            cos_x = np.random.uniform(0.0001, 1.0, num_ions) 
        else:
            cos_x = np.random.uniform(-1.0, 1.0, num_ions)
            
        phi = np.random.uniform(0, 2 * np.pi, num_ions)
        sin_theta = np.sqrt(1 - cos_x**2)
        cos_y = sin_theta * np.cos(phi)
        cos_z = sin_theta * np.sin(phi)
        
        for i in range(num_ions):
            # If coordinates are provided, use them; otherwise default to 0,0,0
            x = coords_df['X_Ang'].iloc[i] if coords_df is not None else 0.0
            y = coords_df['Y_Ang'].iloc[i] if coords_df is not None else 0.0
            z = coords_df['Z_Ang'].iloc[i] if coords_df is not None else 0.0
            
            f.write(f"{ion_z} {ion_mass:.3f} {energy_ev:.1f} {x:.3f} {y:.3f} {z:.3f} {cos_x[i]:.5f} {cos_y[i]:.5f} {cos_z[i]:.5f}\n")

def generate_trim_in(output_path, ion_z, ion_mass, energy_kev, num_ions, title):
    """Writes a perfectly formatted TRIM.IN file matching SRIM-2013 column matrix requirements."""
    
    trim_in_content = f"""==> SRIM-2013.00 This file controls TRIM Calculations.
Ion: Z1 ,  M1,  Energy (keV), Angle,Number,Bragg Corr,AutoSave Number.
    {ion_z:<2}     {ion_mass:<3}        {energy_kev:<8.1f}   0   {num_ions:<5}        1    10000
Cascades(1=No;2=Full;3=Sputt;4-5=Ions;6-7=Neutrons), Random Number Seed, Reminders
                      2                                   0       0
Diskfiles (0=no,1=yes): Ranges, Backscatt, Transmit, Sputtered, Collisions(1=Ion;2=Ion+Recoils), Special EXYZ.txt file
                          0       1           1       0               0                               1
Target material : Number of Elements & Layers
"{title:<39}"       2               2
PlotType (0-5); Plot Depths: Xmin, Xmax(Ang.) [=0 0 for Viewing Full Target]
       0                         0          100020
Target Elements:    Z   Mass(amu)
Atom 1 = Si =      14   28.085
Atom 2 = O  =       8   15.999
Layer   Layer Name /               Width Density    Si(14)    O(8)
Numb.   Description                (Ang) (g/cm3)    Stoich  Stoich
 1      "SiO2"                20    2.32 .333333 .666667
 2      "Silicon"         100000    2.32       1       0
0  Target layer phases (0=Solid, 1=Gas)
0 0 
Target Compound Corrections (Bragg)
 1   1  
Individual target atom displacement energies (eV)
      15      28
Individual target atom lattice binding energies (eV)
       2       3
Individual target atom surface binding energies (eV)
       2       2
Stopping Power Version (1=2011, 0=2011)
 0 
"""
    with open(output_path, 'w') as f:
        f.write(trim_in_content)

def run_srim(srim_dir, work_dir, gen_prefix):
    """Executes TRIM, then copies inputs and moves outputs to preserve generations."""
    print(f"Executing TRIM.exe headlessly for {gen_prefix}...")
    original_dir = os.getcwd()
    
    try:
        os.chdir(srim_dir)
        subprocess.run(["TRIM.exe"])
        print(f"{gen_prefix} simulation complete.")
        
        # 1. Save the input configuration files (Copy)
        for input_file in ["TRIM.IN", "TRIM.DAT"]:
            source = os.path.join(srim_dir, input_file)
            if os.path.exists(source):
                # Formats to something like TRIM_GEN0_Ra224.IN
                file_root, file_ext = input_file.split('.')
                dest_name = f"{file_root}_{gen_prefix}.{file_ext}"
                destination = os.path.join(work_dir, dest_name)
                shutil.copy(source, destination)
                print(f"Saved {dest_name}")

        # 2. Save the output data files (Move)
        for output_file in ["EXYZ.txt", "BACKSCAT.txt"]:
            source = os.path.join(srim_dir, 'SRIM Outputs', output_file)
            if os.path.exists(source):
                # Formats to something like EXYZ_GEN0_Ra224.txt
                file_root, file_ext = output_file.split('.')
                dest_name = f"{file_root}_{gen_prefix}.{file_ext}"
                destination = os.path.join(work_dir, dest_name)
                shutil.move(source, destination)
                print(f"Saved {dest_name}")
                
    finally:
        os.chdir(original_dir)


# ==========================================
# PIPELINE ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    srim_dir = r"C:\Users\lab\Documents\SRIM-2013"
    work_dir = r"C:\Users\lab\Documents\Pablo\RaTag\simulations"
    
    trim_in_path = os.path.join(srim_dir, "TRIM.IN")
    trim_dat_path = os.path.join(srim_dir, "TRIM.DAT")
    
    # ------------------------------------------
    # GEN 0: Th-228 -> Ra-224 Recoil
    # ------------------------------------------
    # print("\n--- Starting Generation 0 (Ra-224) ---")
    # gen0_ions = 10000
    # generate_trim_in(trim_in_path, ion_z=88, ion_mass=224, energy_kev=96.8, num_ions=gen0_ions, title="Ra-224 into SiO2/Si")
    # generate_trim_dat(trim_dat_path, gen0_ions, ion_z=88, ion_mass=224, energy_ev=96800.0, coords_df=None, mode='forward')
    # run_srim(srim_dir, work_dir, gen_prefix="GEN0_Ra224")
    
    # ------------------------------------------
    # GEN 1: Ra-224 -> Rn-220 Recoil
    # ------------------------------------------
    # print("\n--- Starting Generation 1 (Rn-220) ---")
    # coords_gen0 = parse_final_positions(os.path.join(work_dir, "EXYZ_GEN0_Ra224.txt"))
    # gen1_ions = len(coords_gen0)
    # print(f"Implanted ions available for decay: {gen1_ions}")
    
    # generate_trim_in(trim_in_path, ion_z=86, ion_mass=220, energy_kev=103.4, num_ions=gen1_ions, title="Rn-220 from Ra-224")
    # generate_trim_dat(trim_dat_path, gen1_ions, ion_z=86, ion_mass=220, energy_ev=103400.0, coords_df=coords_gen0, mode='full')
    # run_srim(srim_dir, work_dir, gen_prefix="GEN1_Rn220")

    # ------------------------------------------
    # GEN 2: Rn-220 -> Po-216 Recoil
    # ------------------------------------------
    # print("\n--- Starting Generation 2 (Po-216) ---")
    # coords_gen1 = parse_final_positions(os.path.join(work_dir, "EXYZ_GEN1_Rn220.txt"))
    # gen2_ions = len(coords_gen1)
    # print(f"Implanted ions available for decay: {gen2_ions}")
    
    # generate_trim_in(trim_in_path, ion_z=84, ion_mass=216, energy_kev=116.5, num_ions=gen2_ions, title="Po-216 from Rn-220")
    # generate_trim_dat(trim_dat_path, gen2_ions, ion_z=84, ion_mass=216, energy_ev=116500.0, coords_df=coords_gen1, mode='full')
    # run_srim(srim_dir, work_dir, gen_prefix="GEN2_Po216")

    # ------------------------------------------
    # GEN 3: Po-216 -> Pb-212 Recoil
    # ------------------------------------------
    print("\n--- Starting Generation 3 (Pb-212) ---")
    coords_gen2 = parse_final_positions(os.path.join(work_dir, "EXYZ_GEN2_Po216.txt"))
    gen3_ions = len(coords_gen2)
    print(f"Implanted ions available for decay: {gen3_ions}")
    
    generate_trim_in(trim_in_path, ion_z=82, ion_mass=212, energy_kev=127.9, num_ions=gen3_ions, title="Pb-212 from Po-216")
    generate_trim_dat(trim_dat_path, gen3_ions, ion_z=82, ion_mass=212, energy_ev=127900.0, coords_df=coords_gen2, mode='full')
    run_srim(srim_dir, work_dir, gen_prefix="GEN3_Pb212")

    # ------------------------------------------
    # GEN 4 & 5: Pb-212 -> Bi-212 -> Po-212 (Beta Decays)
    # Beta recoils are ~2 eV (below Si displacement threshold).
    # The Po-212 starts exactly where the Pb-212 stopped!
    # ------------------------------------------
    print("\n--- Skipping Beta Decays (Pb-212 & Bi-212) ---")
    print("Recoil energies too low for lattice displacement. Using GEN3 positions.")
    coords_gen3 = parse_final_positions(os.path.join(work_dir, "EXYZ_GEN3_Pb212.txt"))
    
    # ------------------------------------------
    # GEN 6: Po-212 -> Pb-208 Recoil (Main branch)
    # ------------------------------------------
    print("\n--- Starting Generation 6 (Pb-208) ---")
    gen6_ions = len(coords_gen3)
    print(f"Implanted ions available for decay: {gen6_ions}")
    
    generate_trim_in(trim_in_path, ion_z=82, ion_mass=208, energy_kev=170.5, num_ions=gen6_ions, title="Pb-208 from Po-212")
    generate_trim_dat(trim_dat_path, gen6_ions, ion_z=82, ion_mass=208, energy_ev=170500.0, coords_df=coords_gen3, mode='full')
    run_srim(srim_dir, work_dir, gen_prefix="GEN6_Pb208")