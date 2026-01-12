"""
Pipeline for final computation of electron recombination fractions.

This pipeline performs the following steps:
1. Loads X-ray calibration results to obtain the gain factor (g_S2) and W value.
2. Computes the expected number of electrons from the recoil energy and W value.
3. Loads fitted ion S2 areas from previous analyses.
4. Calculates the recombination fraction for each drift field point.
"""

from RaTag.core.datatypes import Run
from RaTag.workflows.electron_recombination import recombination_workflow

def recombination_pipeline(run: Run,
                          path_gs2: str):
    """
    Complete recombination analysis pipeline.
    Args:
        run: Run object with X-ray and ion data
        path_gs2: Path to CSV file with X-ray calibration results
    Returns:
        DataFrame with recombination analysis results
    """
    print("=" * 60)
    print("RECOMBINATION ANALYSIS")
    print("=" * 60)

    df_recomb = recombination_workflow(run, path_gs2)
    return df_recomb
