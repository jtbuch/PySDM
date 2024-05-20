import numpy as np
from typing import Dict
from PySDM.initialisation.impl.spectrum import Spectrum
from .common_bimodal import Common

from PySDM.physics import si


class StratoCumulus(Common):
    def __init__(
        self,
        formulae,
        rhod_w_max: float,
        n_sd_per_mode: tuple,
        aerosol_modes_by_kappa: Dict[float, Spectrum],
    ):
        super().__init__(formulae, n_sd_per_mode, aerosol_modes_by_kappa)
        self.th_std0 = 289 * si.kelvins
        self.initial_water_vapour_mixing_ratio = 7.5 * si.grams / si.kilogram
        self.p0 = 1015 * si.hectopascals
        self.rhod_w_max = rhod_w_max

    def stream_function(self, xX, zZ, _):
        X = self.size[0]
        return (
            -self.rhod_w_max * X / np.pi * np.sin(np.pi * zZ) * np.cos(2 * np.pi * xX)
        )

    def rhod_of_zZ(self, zZ):
        p = getattr(
            self.formulae.hydrostatics,
            "p_of_z_assuming_const_th_and_initial_water_vapour_mixing_ratio",
        )(
            self.formulae.constants.g_std,
            self.p0,
            self.th_std0,
            self.initial_water_vapour_mixing_ratio,
            z=zZ * self.size[-1],
        )
        rhod = self.formulae.state_variable_triplet.rho_d(
            p, self.initial_water_vapour_mixing_ratio, self.th_std0
        )
        return rhod
