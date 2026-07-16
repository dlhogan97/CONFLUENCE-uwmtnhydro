import numpy as np

def q_to_e0(q, p):
    MV_CST = 0.622
    e_0 = (q * p) / (MV_CST + q * (1 - MV_CST))
    return e_0

def empirical_lw_dilley_obrien(Tair, p, q):
    """
    Calculate incoming longwave radiation using the Dilley and O'Brien (1998) empirical formula.

    Parameters:
    Tair : float or np.array
        Air temperature in degrees Kelvin.
    p : float or np.array
        Atmospheric pressure in Pa.
    q : float or np.array
        Specific humidity.

    Returns:
    LWin : float or np.array
        Incoming longwave radiation in W/m².
    """
    CONSTANT_1 = 59.38
    CONSTANT_2 = 113.7
    KELVIN_OFFSET = 273.16
    CONSTANT_3 = 96.96
    CONSTANT_4 = 4650
    # Dilley & O'Brien (1998) is 96.96*sqrt(w/25) with precipitable water
    # w [kg m-2] = 4650*e0/Tair (e0 in kPa).  A divisor of 2.5 inflates the
    # vapour term by sqrt(10) and drives effective emissivity above 1.0.
    CONSTANT_5 = 25.0

    e_0 = q_to_e0(q, p/1000)  # Calculate actual vapor pressure

    LWin = CONSTANT_1 + (CONSTANT_2 * (Tair/KELVIN_OFFSET)**6) + (CONSTANT_3 * np.sqrt((CONSTANT_4 * e_0)/(CONSTANT_5*Tair)))
    return LWin