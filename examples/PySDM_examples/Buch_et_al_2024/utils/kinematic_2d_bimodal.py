"""
Two-dimensional single-eddy prescribed-flow framework with moisture and heat advection
handled by [PyMPDATA](http://github.com/open-atmos/PyMPDATA/)
"""

import numpy as np

from PySDM.environments.impl.moist import Moist
from PySDM.impl.mesh import Mesh
from PySDM.initialisation.equilibrate_wet_radii import (
    default_rtol,
    equilibrate_wet_radii,
)
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.impl import arakawa_c
from PySDM.environments.impl import register_environment


@register_environment()
class Kinematic2D(Moist):
    def __init__(self, *, dt, grid, size, rhod_of, mixed_phase=False):
        super().__init__(dt, Mesh(grid, size), [], mixed_phase=mixed_phase)
        self.rhod_of = rhod_of
        self.formulae = None

    def register(self, builder):
        super().register(builder)
        self.formulae = builder.particulator.formulae
        rhod = builder.particulator.Storage.from_ndarray(
            arakawa_c.make_rhod(self.mesh.grid, self.rhod_of).ravel()
        )
        self._values["current"]["rhod"] = rhod
        self._tmp["rhod"] = rhod

    @property
    def dv(self):
        return self.mesh.dv

    def init_attributes(
        self,
        *,
        spatial_discretisation,
        rtol=default_rtol,
        n_sd_per_mode,
        aerosol_modes_by_kappa,
        z_part=None,
        x_part=None
    ):
        super().sync()
        self.notify()
        # n_sd = n_sd or self.particulator.n_sd

        attributes = {
            k: np.empty(0)
            for k in (
                "cell id",
                "cell origin",
                "position in cell",
                "dry volume",
                "kappa times dry volume",
                "multiplicity",
                "volume",
            )
        }
        rhod = self["rhod"].to_ndarray()
        domain_volume = np.prod(np.array(self.mesh.size))

        with np.errstate(all="raise"):
            for i, (kappa, spectrum) in enumerate(aerosol_modes_by_kappa.items()):
                positions = spatial_discretisation.sample(
                    backend=self.particulator.backend,
                    grid=self.mesh.grid,
                    n_sd=n_sd_per_mode[i] * self.mesh.grid[0] * self.mesh.grid[1],
                    z_part=z_part[i],
                    x_part=x_part[i],
                )

                cell_id, cell_origin, pos_cell = self.mesh.cellular_attributes(
                    positions
                )
                attributes["cell id"] = np.append(attributes["cell id"], cell_id)
                if i == 0:
                    attributes["cell origin"] = cell_origin
                    attributes["position in cell"] = pos_cell
                else:
                    attributes["cell origin"] = np.append(
                        attributes["cell origin"], cell_origin, axis=1
                    )
                    attributes["position in cell"] = np.append(
                        attributes["position in cell"], pos_cell, axis=1
                    )

                sampling = ConstantMultiplicity(spectrum)
                r_dry, n_per_kg = sampling.sample(
                    backend=self.particulator.backend,
                    n_sd=n_sd_per_mode[i] * self.mesh.grid[0] * self.mesh.grid[1],
                )
                v_dry = self.formulae.trivia.volume(radius=r_dry)
                attributes["dry volume"] = np.append(attributes["dry volume"], v_dry)
                attributes["kappa times dry volume"] = np.append(
                    attributes["kappa times dry volume"], v_dry * kappa
                )
                attributes["multiplicity"] = np.append(
                    attributes["multiplicity"], n_per_kg * rhod[cell_id] * domain_volume
                )

                r_wet = equilibrate_wet_radii(
                    r_dry=v_dry,
                    environment=self,
                    cell_id=cell_id,
                    kappa_times_dry_volume=v_dry * kappa,
                    rtol=rtol,
                )
                attributes["volume"] = np.append(
                    attributes["volume"], self.formulae.trivia.volume(radius=r_wet)
                )

            attributes["cell id"] = np.array(attributes["cell id"], dtype=int)
            attributes["cell origin"] = np.array(attributes["cell origin"], dtype=int)

        return attributes

    def get_thd(self):
        return self.particulator.dynamics["EulerianAdvection"].solvers["th"]

    def get_water_vapour_mixing_ratio(self):
        return self.particulator.dynamics["EulerianAdvection"].solvers[
            "water_vapour_mixing_ratio"
        ]

    def sync(self):
        self.particulator.dynamics["EulerianAdvection"].solvers.wait()
        super().sync()
