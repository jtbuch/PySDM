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
                attributes["cell origin"]= np.array([np.append(attributes["cell origin"], cell_origin)])
                attributes["position in cell"]= np.array([np.append(attributes["position in cell"], pos_cell)])

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

    def inject_particles(
        self,
        *,
        spatial_discretisation,
        z_part=None,
        t_dur= None,
        n_seed_sds=None,
        kappa=None,
        r_dry=None,
        m_param=None
    ):
        super().sync()
        self.notify()

        with np.errstate(all="raise"):
            positions = spatial_discretisation.sample(
                    backend=self.particulator.backend,
                    grid=self.mesh.grid,
                    n_sd= n_seed_sds*t_dur,
                    z_part=z_part,
                ) 
            p_indx= np.random.choice(np.arange(n_seed_sds*t_dur), n_seed_sds)
            cell_id, cell_origin, pos_cell= self.mesh.cellular_attributes(np.array([positions[0, p_indx]])) 
            self.particulator.attributes["cell id"].data= np.append(self.particulator.attributes["cell id"].data, cell_id)
            self.particulator.attributes["cell origin"].data= np.array([np.append(self.particulator.attributes["cell origin"].data, cell_origin)])
            self.particulator.attributes["position in cell"].data= np.array([np.append(self.particulator.attributes["position in cell"].data, pos_cell)])
            
            r_dry= np.ones(n_seed_sds)*r_dry
            v_dry= self.formulae.trivia.volume(radius=r_dry)
            self.particulator.attributes["dry volume"].data= np.append(self.particulator.attributes["dry volume"].data, v_dry)
            self.particulator.attributes["kappa times dry volume"].data= np.append(self.particulator.attributes["kappa times dry volume"].data, v_dry * kappa)
            self.particulator.attributes["multiplicity"].data= np.append(self.particulator.attributes["multiplicity"].data, np.ones(n_seed_sds)*m_param)

            r_wet = equilibrate_wet_radii(
                    r_dry= v_dry,
                    environment=self,
                    cell_id= cell_id,
                    kappa_times_dry_volume= v_dry * kappa,
                    )
            self.particulator.attributes["volume"].data= np.append(self.particulator.attributes["volume"].data, self.formulae.trivia.volume(radius=r_wet))

            self.particulator.attributes["cell id"].data= np.array(self.particulator.attributes["cell id"].data, dtype= int)
            self.particulator.attributes["cell origin"].data= np.array(self.particulator.attributes["cell origin"].data, dtype= int)

            self.particulator.attributes["water mass"].data[-n_seed_sds:]= \
                            self.particulator.formulae.particle_shape_and_density.volume_to_mass(self.particulator.attributes["volume"].data[-n_seed_sds:])


            return self.particulator.attributes

    @property
    def dv(self):
        return self.mesh.dv
