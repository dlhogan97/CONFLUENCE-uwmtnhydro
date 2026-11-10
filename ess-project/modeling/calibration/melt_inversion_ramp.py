"""
Elevation-ramped albedo decay (2026-10-27): variant D's compressed albedo range (floor vis 0.62 / NIR 0.48, fresh max 0.88 / 0.62) with albedoDecayRate
ramped log-linearly in elevation, fast at low elevation and slow up high, because Dana (3040 m HRU) preferred slow decay and Tuolumne Meadows
(2409-2730 m HRUs) fast decay. Same years, HRUs and metrics as melt_inversion_tests.py; A and D rerun for reference.
usage: python melt_inversion_ramp.py
"""
import numpy as np
import deepstore_build as DB
import melt_inversion_tests as T

D = dict(albMinVis=0.62, albMinNIR=0.48, maxVis=0.88, maxNIR=0.62)
T.VARIANTS = {"A base": {}, "D B + max 0.88/0.62": dict(D),
              "G D + decay ramp 6e5@2409 -> 1.29e6@3040": dict(D, ramp=(2409.0, 6e5, 3040.0, 1.29e6)),
              "GO G + open canopy": dict(D, ramp=(2409.0, 6e5, 3040.0, 1.29e6), open=True),
              "H D + decay ramp 4e5@2409 -> 1.6e6@3040": dict(D, ramp=(2409.0, 4e5, 3040.0, 1.6e6)),
              "HO H + open canopy": dict(D, ramp=(2409.0, 4e5, 3040.0, 1.6e6), open=True)}
T.OUT = T.I.OUT/"tests_ramp"; T.OUT.mkdir(exist_ok=True)
_base_apply = T.Runner._apply_params


def _apply(self, work, params):
    _base_apply(self, work, params); rp = self.variant.get("ramp")
    if rp:
        z0, d0, z1, d1 = rp; w = np.clip((self.elev - z0)/(z1 - z0), 0.0, 1.0)
        DB.set_trial_param(work, "albedoDecayRate", 10**(np.log10(d0) + w*(np.log10(d1) - np.log10(d0))))


T.Runner._apply_params = _apply

if __name__ == "__main__":
    T.main()
