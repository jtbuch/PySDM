from typing import Dict, Iterable, Optional
from PySDM.initialisation.impl.spectrum import Spectrum
import numpy as np
from utils.strato_cumulus_bimodal import StratoCumulus

from PySDM import Formulae
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import Gaussian
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.physics import si


class Settings(StratoCumulus):
    def __dir__(self) -> Iterable[str]:
        return (
            "dt",
            "grid",
            "size",
            "n_spin_up",
            "versions",
            "steps_per_output_interval",
            "formulae",
            "initial_dry_potential_temperature_profile",
            "initial_vapour_mixing_ratio_profile",
            "rhod_w_max",
            "n_sd_per_mode",
            "aerosol_modes_by_kappa",
            "z_part",
            "x_part",
            "simulation_time",
            "spin_up_time",
            "r_seed",
            "m_param",
            "kappa_seed",
        )

    def __init__(
        self,
        formulae=None,
        rhod_w_max: float = 0.6 * si.metres / si.seconds * (si.kilogram / si.metre**3),
        grid: tuple = None,
        size: tuple = None,
        n_sd_per_mode: tuple = None,
        aerosol_modes_by_kappa: Dict[float, Spectrum] = None,
        z_part: Optional[tuple] = None,
        x_part: Optional[tuple] = None,
        simulation_time: float = None,
        dt: float = None,
        spin_up_time: float = None,
        r_seed: float = None,
        m_param: float = None,
        kappa_seed: float = None,
    ):
        super().__init__(
            formulae or Formulae(),
            rhod_w_max=rhod_w_max,
            n_sd_per_mode=n_sd_per_mode,
            aerosol_modes_by_kappa=aerosol_modes_by_kappa,
        )

        self.grid = grid
        self.size = size
        self.z_part = z_part
        self.x_part = x_part

        # output steps
        self.simulation_time = simulation_time  # 90 * si.minute
        self.dt = dt  # 5 * si.second
        self.spin_up_time = spin_up_time  # 1 * si.hour

        # seed properties
        self.r_seed = r_seed
        self.m_param = m_param
        self.kappa_seed = kappa_seed

        # additional breakup dynamics
        mu_r = 10 * si.um
        mu = 4 / 3 * np.pi * mu_r**3
        sigma = mu / 2.5
        vmin = mu / 1000
        self.coalescence_efficiency = ConstEc(Ec=0.95)
        self.breakup_efficiency = ConstEb(Eb=1.0)
        self.breakup_fragmentation = Gaussian(mu=mu, sigma=sigma, vmin=vmin, nfmax=10)
