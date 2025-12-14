"""
Quick test of energy calibration functionality.
"""
import numpy as np
from RaTag.alphas.spectrum_fitting import EnergyCalibration, derive_energy_calibration

def test_energy_calibration_basic():
    """Test basic EnergyCalibration functionality."""
    # Create simple calibration
    cal = EnergyCalibration(
        a=1.2,
        b=-0.5,
        anchors={'Th228': (4.6, 5.4), 'Ra224': (4.8, 5.7)},
        residuals=0.01
    )
    
    # Test apply
    E_SCA = np.array([4.6, 4.8, 5.0])
    E_true = cal.apply(E_SCA)
    
    print("Applied calibration:")
    print(f"  E_SCA = {E_SCA}")
    print(f"  E_true = {E_true}")
    
    # Test inverse
    E_SCA_recovered = cal.inverse(E_true)
    print(f"  E_SCA (recovered) = {E_SCA_recovered}")
    
    # Check round-trip
    assert np.allclose(E_SCA, E_SCA_recovered), "Round-trip failed!"
    
    print("✓ EnergyCalibration basic test passed")
    

if __name__ == "__main__":
    test_energy_calibration_basic()
