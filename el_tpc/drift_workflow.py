from dataclasses import replace
from typing import cast
from RaTag.core.datatypes import Run, SetPmt


from dataclasses import replace
from RaTag.core import units
from RaTag.core.functional import map_over
from RaTag.core.decorators import disk_cache, require_attributes
from RaTag.el_tpc import physics

@require_attributes('pressure', 'temperature')
def with_gas_density(run: Run) -> Run:
    gd = physics.gas_density_cm3(run.pressure, run.temperature)
    return replace(run, gas_density=gd)

@disk_cache(target_attr='drift_field')
@require_attributes('gate', 'anode')
def set_fields(set_pmt: SetPmt, el_gap_cm: float, drift_gap_cm: float, force: bool = False) -> SetPmt:
    """
    Calculates the actual and reduced fields. 
    """
    
    v_gate = cast(float, units.V(set_pmt.gate)) #type: ignore # Ensure we have a float for calculations
    v_anode = cast(float, units.V(set_pmt.anode)) #type:ignore  # Ensure we have a float for calculations
    
    drift_field = v_gate / drift_gap_cm 
    el_field = (v_anode - v_gate) / el_gap_cm

    return replace(set_pmt, 
                   drift_field = drift_field,
                   EL_field = el_field)

@disk_cache(target_attr='red_EL_field')
@require_attributes('drift_field', 'EL_field')
def set_reduced_fields(set_pmt: SetPmt, gas_density_cm3: float, force: bool = False) -> SetPmt:
    """
        Calculates reduced fields from actual fields and gas density.
    """

    E_drift = cast(float, set_pmt.drift_field)  # Ensure we have a float for calculations
    E_EL = cast(float, set_pmt.EL_field) 
    red_drift_Vcm2 = physics.compute_reduced_field(E_drift, gas_density_cm3)
    red_drift_Td = units.to_Td(red_drift_Vcm2)  # Convert to Td

    red_el_Vcm2 = physics.compute_reduced_field(E_EL, gas_density_cm3)
    red_el_Td = units.to_Td(red_el_Vcm2)  # Convert to Td

    return replace(set_pmt, 
                   red_drift_field=red_drift_Td,
                   red_EL_field=red_el_Td)

@disk_cache(target_attr='time_drift')
@require_attributes('red_drift_field')
def set_transport(set_pmt: SetPmt, drift_gap_cm: float, force: bool = False) -> SetPmt:
    """
        Calculates drift velocities and times. 
    """        
    redE_drift = cast(float, set_pmt.red_drift_field)  # Ensure we have a float for calculations
    # Calculate drift speed from reduced field
    speed_mmus = physics.redfield_to_speed(redE_drift) # returns mm/us
    
    # Calculate time
    drift_gap_mm = units.cm_to_mm(drift_gap_cm)
    time_drift_us = drift_gap_mm / speed_mmus if speed_mmus else None # The decorator should prevent this from being None, but redfield_to_speed could return None if the model fails, so we keep this for redundancy.
    
    # diffusion = transport.redfield_to_diffusion(set_pmt.red_drift_field) # TODO: add diffusion model and calculation here
    diffusion = None
    return replace(set_pmt, 
                   speed_drift=speed_mmus,
                   time_drift=time_drift_us,
                   diffusion_coefficient=diffusion)


def resolve_set_drift(set_pmt: SetPmt, run: Run, force: bool = False) -> SetPmt:
    """Calculate fields and transport properties for a single set, given the run parameters."""
    # 1. Unpack & Math

    set_with_fields = set_fields(set_pmt, drift_gap_cm=run.drift_gap, el_gap_cm=run.el_gap, force=force)
    set_with_red_fields = set_reduced_fields(set_with_fields, run.gas_density, force=force)
    set_with_transport = set_transport(set_with_red_fields, drift_gap_cm=run.drift_gap, force=force)
    
    # 2. Repack 
    return set_with_transport


def map_drift_physics(run: Run, force: bool = False) -> Run:
    """Explicit, flat pipeline using safe FP mapping."""
    
    run_with_density = with_gas_density(run)
    bound_func = lambda s: resolve_set_drift(s, run_with_density, force=force)
    
    # map_over safely handles the loop and the try/except blocks
    enriched_sets = map_over(run_with_density.sets, bound_func, catch_errors=True)
    
    return replace(run_with_density, sets=enriched_sets)