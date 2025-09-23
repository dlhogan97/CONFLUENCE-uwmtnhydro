import numpy as np

def q_to_e0(q, p):
    MV_CST = 0.622
    e_0 = (q * p) / (MV_CST + q * (1 - MV_CST))
    return e_0
def emperical_lw_dilley_obrien(Tair, p, q):
    """
    Calculate incoming longwave radiation using the Dilley and O'Brien (1998) empirical formula.

    Parameters:
    Tair : float or np.array
        Air temperature in degrees Celsius.
    e_0 : float or np.array
        Actual vapor pressure in kPa.
        
    Returns:
    LWin : float or np.array
        Incoming longwave radiation in W/m².
    """
    CONSTANT_1 = 59.38
    CONSTANT_2 = 113.7
    KELVIN_OFFSET = 273.16
    CONSTANT_3 = 96.96
    CONSTANT_4 = 465
    CONSTANT_5 = 2.5

    e_0 = q_to_e0(q, p)  # Calculate actual vapor pressure

    LWin = CONSTANT_1 + (CONSTANT_2 * (Tair/KELVIN_OFFSET)**6) + (CONSTANT_3 * np.sqrt((CONSTANT_4 * 1000 * e_0)/(CONSTANT_5*Tair)))
    return LWin