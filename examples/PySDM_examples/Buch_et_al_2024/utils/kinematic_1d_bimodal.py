"""
Single-column time-varying-updraft framework with moisture advection handled by
[PyMPDATA](http://github.com/open-atmos/PyMPDATA/)
"""

import numpy as np

from PySDM.environments.impl.moist import Moist

from PySDM.impl import arakawa_c
from PySDM import Formulae
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.initialisation.equilibrate_wet_radii import equilibrate_wet_radii


class Kinematic1D(Moist):
    def __init__(self, *, dt, mesh, thd_of_z, rhod_of_z, z0=0):
        super().__init__(dt, mesh, [])
        self.thd0 = thd_of_z(z0 + mesh.dz * arakawa_c.z_scalar_coord(mesh.grid))
        self.rhod = rhod_of_z(z0 + mesh.dz * arakawa_c.z_scalar_coord(mesh.grid))
        self.formulae = None

    def register(self, builder):
        super().register(builder)
        self.formulae = builder.particulator.formulae
        rhod = builder.particulator.Storage.from_ndarray(self.rhod)
        self._values["current"]["rhod"] = rhod
        self._tmp["rhod"] = rhod

    def get_water_vapour_mixing_ratio(self) -> np.ndarray:
        return self.particulator.dynamics["EulerianAdvection"].solvers.advectee.get()

    def get_thd(self) -> np.ndarray:
        return self.thd0

    def init_attributes(
        self,
        *,
        spatial_discretisation,
        n_sd_per_mode, 
        nz_tot, 
        aerosol_modes_by_kappa,
        z_part=None,
        collisions_only=False
    ):
        super().sync()
        self.notify()

        attributes = {k: np.empty(0) for k in ("cell id", "cell origin", "position in cell", "dry volume", "kappa times dry volume", "multiplicity", "volume")}
        rhod = self["rhod"].to_ndarray()
        domain_volume = np.prod(np.array(self.mesh.size))

        with np.errstate(all="raise"):
            for i, (kappa, spectrum) in enumerate(aerosol_modes_by_kappa.items()):
                positions = spatial_discretisation.sample(
                    backend=self.particulator.backend,
                    grid=self.mesh.grid,
                    n_sd= n_sd_per_mode[i]*nz_tot,
                    z_part=z_part[i],
                ) #self.particulator.n_sd
                
                cell_id, cell_origin, pos_cell= self.mesh.cellular_attributes(positions)
                attributes["cell id"]= np.append(attributes["cell id"], cell_id)
                attributes["cell origin"]= np.append(attributes["cell origin"], cell_origin)
                attributes["position in cell"]= np.append(attributes["position in cell"], pos_cell)

                if collisions_only:
                    sampling = ConstantMultiplicity(spectrum)
                    v_wet, n_per_kg = sampling.sample(
                        backend=self.particulator.backend, n_sd= n_sd_per_mode[i]*nz_tot
                    )
                    # attributes["dry volume"] = v_wet
                    attributes["volume"] = v_wet
                    # attributes["kappa times dry volume"] = attributes["dry volume"] * kappa
                else:
                    sampling = ConstantMultiplicity(spectrum)
                    r_dry, n_per_kg = sampling.sample(
                        backend=self.particulator.backend, n_sd= n_sd_per_mode[i]*nz_tot # include zpart for seeded aerosols
                    )
                    v_dry = self.formulae.trivia.volume(radius=r_dry)
                    attributes["dry volume"] = np.append(attributes["dry volume"], v_dry)
                    attributes["kappa times dry volume"] = np.append(attributes["kappa times dry volume"], v_dry * kappa)
                    attributes["multiplicity"]= np.append(attributes["multiplicity"], n_per_kg * rhod[cell_id] * domain_volume)

                    r_wet = equilibrate_wet_radii(
                        r_dry= v_dry,
                        environment=self,
                        cell_id= cell_id,
                        kappa_times_dry_volume= v_dry * kappa,
                        )
                    attributes["volume"]= np.append(attributes["volume"], self.formulae.trivia.volume(radius=r_wet))

            attributes["cell id"]= np.array(attributes["cell id"], dtype= int)
            attributes["cell origin"]= np.array(attributes["cell origin"], dtype= int)

        return attributes

    @property
    def dv(self):
        return self.mesh.dv
