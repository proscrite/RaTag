from dataclasses import replace

from .constructors import populate_run, set_fields, set_transport_properties, estimate_s1_from_batches
from .analysis import integrate_run_s2, fit_run_s2
from .dataIO import store_s2area
from .datatypes import Run, SetPmt
from .config import IntegrationConfig, FitConfig
from .transport import with_gas_density

def prepare_set(s: SetPmt, run: Run) -> SetPmt:
    s1 = estimate_s1_from_batches(s, n_batches=5, batch_size=20)
    s1 = set_fields(s1, drift_gap_cm=run.drift_gap, el_gap_cm=run.el_gap, gas_density=run.gas_density)
    s1 = set_transport_properties(s1, drift_gap_cm=run.drift_gap, transport=None)
    return s1


# Build run (no logic inside Run class itself!)
run5 = Run(root_directory = base_dir,
           run_id = "RUN5",
            el_field = 2375,            # V/cm
            target_isotope = "Th228",
            pressure = 2.0,             # bar
            temperature = 297,          # K
            sampling_rate = 5e9,
            el_gap = 0.8,               # cm
            drift_gap = 1.4,            # cm
            width_s2 = 5.6              # in µs    
        )

run5 = with_gas_density(run5) # Add gas density
# Load sets
run5 = populate_run(run5)
# Prepare sets
run5 = replace(run5, sets=[prepare_set(s, run5) for s in run5.sets])


# Integrate and fit
areas = integrate_run_s2(run5, integration_config=IntegrationConfig())
fitted = fit_run_s2(areas, fit_config=FitConfig())

# Store results
for s2 in fitted.values():
    store_s2area(s2)
