from collections import namedtuple

import numpy as np
from PySDM_examples.Shipway_and_Hill_2012.mpdata_1d import MPDATA_1D

import PySDM.products as PySDM_products
from PySDM.builder import Builder
from PySDM.backends import CPU
from PySDM.dynamics import (
    AmbientThermodynamics,
    Condensation,
    Coalescence,
    Displacement,
    EulerianAdvection,
)
from utils.kinematic_1d_bimodal import Kinematic1D
from PySDM.impl.mesh import Mesh
from PySDM.initialisation.sampling import spectral_sampling
from PySDM.initialisation.sampling.spatial_sampling import Pseudorandom
from PySDM.physics import si


class Simulation:
    def __init__(self, settings, backend=CPU):
        self.nt = settings.nt
        self.nz = settings.nz
        self.z0 = -settings.particle_reservoir_depth
        self.save_spec_and_attr_times = settings.save_spec_and_attr_times
        self.number_of_bins = settings.number_of_bins

        self.particulator = None
        self.output_attributes = None
        self.output_products = None
        self.n_seed_sds = settings.n_seed_sds
        self.r_seed = settings.r_seed
        self.kappa_seed = settings.kappa_seed
        self.m_param = settings.m_param
        self.seed_z_part = settings.seed_z_part
        self.seed_step = int(settings.t_part[1] / settings.dt)

        self.mesh = Mesh(
            grid=(settings.nz,),
            size=(settings.z_max + settings.particle_reservoir_depth,),
        )

        self.env = Kinematic1D(
            dt=settings.dt,
            mesh=self.mesh,
            thd_of_z=settings.thd,
            rhod_of_z=settings.rhod,
            z0=-settings.particle_reservoir_depth,
        )

        def zZ_to_z_above_reservoir(zZ):
            z_above_reservoir = zZ * (settings.nz * settings.dz) + self.z0
            return z_above_reservoir

        self.mpdata = MPDATA_1D(
            nz=settings.nz,
            dt=settings.dt,
            mpdata_settings=settings.mpdata_settings,
            advector_of_t=lambda t: settings.rho_times_w(t) * settings.dt / settings.dz,
            advectee_of_zZ_at_t0=lambda zZ: settings.water_vapour_mixing_ratio(
                zZ_to_z_above_reservoir(zZ)
            ),
            g_factor_of_zZ=lambda zZ: settings.rhod(zZ_to_z_above_reservoir(zZ)),
        )

        _extra_nz = settings.particle_reservoir_depth // settings.dz
        _z_vec = settings.dz * np.linspace(
            -_extra_nz, settings.nz - _extra_nz, settings.nz + 1
        )
        self.g_factor_vec = settings.rhod(_z_vec)

        self.builder = Builder(
            n_sd=settings.n_sd,
            backend=backend(formulae=settings.formulae),
            environment=self.env,
        )
        self.builder.add_dynamic(AmbientThermodynamics())

        if settings.enable_condensation:
            self.builder.add_dynamic(
                Condensation(
                    adaptive=settings.condensation_adaptive,
                    rtol_thd=settings.condensation_rtol_thd,
                    rtol_x=settings.condensation_rtol_x,
                    update_thd=settings.condensation_update_thd,
                )
            )
        self.builder.add_dynamic(EulerianAdvection(self.mpdata))

        self.products = []
        if settings.precip:
            self.add_collision_dynamic(self.builder, settings, self.products)

        displacement = Displacement(
            enable_sedimentation=settings.precip,
            precipitation_counting_level_index=int(
                settings.particle_reservoir_depth / settings.dz
            ),
        )
        self.builder.add_dynamic(displacement)

        # Moving spectral sampling by components to kinematic_1d_bimodal.py
        self.attributes = self.env.init_attributes(
            spatial_discretisation=Pseudorandom(),
            n_sd_per_mode=settings.n_sd_per_mode,
            nz_tot=settings.nz,
            aerosol_modes_by_kappa=settings.aerosol_modes_by_kappa,
            collisions_only=not settings.enable_condensation,
            z_part=settings.z_part,
        )
        self.products += [
            PySDM_products.WaterMixingRatio(
                name="cloud water mixing ratio",
                unit="g/kg",
                radius_range=settings.cloud_water_radius_range,
            ),
            PySDM_products.WaterMixingRatio(
                name="rain water mixing ratio",
                unit="g/kg",
                radius_range=settings.rain_water_radius_range,
            ),
            PySDM_products.AmbientDryAirDensity(name="rhod"),
            PySDM_products.AmbientDryAirPotentialTemperature(name="thd"),
            PySDM_products.ParticleSizeSpectrumPerVolume(
                name="wet spectrum", radius_bins_edges=settings.r_bins_edges
            ),
            PySDM_products.ParticleConcentration(
                name="nc", radius_range=settings.cloud_water_radius_range
            ),
            PySDM_products.ParticleConcentration(
                name="nr", radius_range=settings.rain_water_radius_range
            ),
            PySDM_products.ParticleConcentration(
                name="na", radius_range=(0, settings.cloud_water_radius_range[0])
            ),
            PySDM_products.MeanRadius(),
            PySDM_products.EffectiveRadius(
                radius_range=settings.cloud_water_radius_range
            ),
            PySDM_products.SuperDropletCountPerGridbox(),
            PySDM_products.NumberSizeSpectrum(
                name="N(v)", radius_bins_edges=settings.r_bins_edges
            ),
            PySDM_products.ParticleVolumeVersusRadiusLogarithmSpectrum(
                name="dvdlnr", radius_bins_edges=settings.r_bins_edges
            ),
        ]
        if settings.enable_condensation:
            self.products.extend(
                [
                    PySDM_products.RipeningRate(name="ripening"),
                    PySDM_products.ActivatingRate(name="activating"),
                    PySDM_products.DeactivatingRate(name="deactivating"),
                    PySDM_products.PeakSupersaturation(unit="%"),
                    PySDM_products.ParticleSizeSpectrumPerVolume(
                        name="dry spectrum",
                        radius_bins_edges=settings.r_bins_edges_dry,
                        dry=True,
                    ),
                ]
            )
        if settings.precip:
            self.products.extend(
                [
                    PySDM_products.CollisionRatePerGridbox(
                        name="collision_rate",
                    ),
                    PySDM_products.CollisionRateDeficitPerGridbox(
                        name="collision_deficit",
                    ),
                    PySDM_products.CoalescenceRatePerGridbox(
                        name="coalescence_rate",
                    ),
                ]
            )
        self.particulator = self.builder.build(
            attributes=self.attributes, products=tuple(self.products)
        )

        self.output_attributes = {
            "cell origin": [],
            "position in cell": [],
            "radius": [],
            "multiplicity": [],
        }
        self.output_products = {}
        for k, v in self.particulator.products.items():
            if len(v.shape) == 1:
                self.output_products[k] = np.zeros((self.mesh.grid[-1], self.nt + 1))
            elif len(v.shape) == 2:
                number_of_time_sections = len(self.save_spec_and_attr_times)
                self.output_products[k] = np.zeros(
                    (self.mesh.grid[-1], self.number_of_bins, number_of_time_sections)
                )
        self.output_products["t"] = np.linspace(
            0, self.nt * self.particulator.dt, self.nt + 1, endpoint=True
        )
        self.output_products["z"] = np.linspace(
            self.z0 + self.mesh.dz / 2,
            self.z0 + (self.mesh.grid[-1] - 1 / 2) * self.mesh.dz,
            self.mesh.grid[-1],
            endpoint=True,
        )

    @staticmethod
    def add_collision_dynamic(builder, settings, _):
        builder.add_dynamic(
            Coalescence(
                collision_kernel=settings.collision_kernel,
                adaptive=settings.coalescence_adaptive,
            )
        )

    def save_scalar(self, step):
        for k, v in self.particulator.products.items():
            if len(v.shape) > 1:
                continue
            self.output_products[k][:, step] = v.get()

    def save_spectrum(self, index):
        for k, v in self.particulator.products.items():
            if len(v.shape) == 2:
                self.output_products[k][:, :, index] = v.get()

    def save_attributes(self):
        for k, v in self.output_attributes.items():
            v.append(self.particulator.attributes[k].to_ndarray())

    def save(self, step):
        self.save_scalar(step)
        time = step * self.particulator.dt
        if len(self.save_spec_and_attr_times) > 0 and (
            np.min(
                np.abs(
                    np.ones_like(self.save_spec_and_attr_times) * time
                    - np.array(self.save_spec_and_attr_times)
                )
            )
            < 0.1
        ):
            save_index = np.argmin(
                np.abs(
                    np.ones_like(self.save_spec_and_attr_times) * time
                    - np.array(self.save_spec_and_attr_times)
                )
            )
            self.save_spectrum(save_index)
            self.save_attributes()

    def run(self):
        for step in range(self.nt - self.particulator.n_steps):
            self.mpdata.update_advector_field()
            if "Displacement" in self.particulator.dynamics:
                self.particulator.dynamics["Displacement"].upload_courant_field(
                    (self.mpdata.advector / self.g_factor_vec,)
                )
            self.particulator.run(steps=1)
            self.save(step + 1)

        Outputs = namedtuple("Outputs", "products attributes")
        output_results = Outputs(self.output_products, self.output_attributes)
        return output_results

    def stepwise_sd_update(self, seed_step):

        cell_edge_arr = np.linspace(
            self.particulator.attributes["position in cell"].data[0, :].min(),
            self.particulator.attributes["position in cell"].data[0, :].max(),
            self.nz,
        )
        ncell_arr = (
            np.digitize(
                self.particulator.attributes["position in cell"].data[0, :],
                cell_edge_arr,
            )
            - 1
        )
        self.save(0)

        for i in range(self.nt):

            if i in seed_step:
                try:
                    # ramdomly select a SD candidate to be the potential seed; tolerance set to half the seed radius
                    potseed_arr = np.where(
                        np.abs(
                            self.particulator.attributes["radius"].data - self.r_seed
                        )
                        < self.r_seed / 2
                    )[0]
                    potindx_arr = np.where(
                        (
                            self.particulator.attributes["position in cell"].data[
                                0, potseed_arr
                            ]
                            > self.seed_z_part[0]
                        )
                        & (
                            self.particulator.attributes["position in cell"].data[
                                0, potseed_arr
                            ]
                            <= self.seed_z_part[1]
                        )
                    )[0]
                    potseed = np.random.choice(potseed_arr[potindx_arr], 1)[0]

                    # find all SDs in the same cell as the potential seed
                    npotseed_arr = np.where(ncell_arr == ncell_arr[potseed])[0]
                    npotseed_arr = npotseed_arr[npotseed_arr != potseed]

                    # update the attributes of the potential seed and the other SDs in the same cell
                    # if multiplicity of all neighboring SDs is increased, there is a distinct seeding signal
                    gamma_fac = (
                        self.particulator.attributes["multiplicity"].data[potseed]
                        / self.particulator.attributes["multiplicity"].data[
                            npotseed_arr
                        ]
                    ).astype(int)
                    gamma_fac[gamma_fac == 0] = 1
                    self.particulator.attributes["multiplicity"].data[
                        potseed
                    ] = self.m_param
                    self.particulator.attributes["kappa times dry volume"].data[
                        potseed
                    ] = (
                        self.particulator.attributes["dry volume"].data[potseed]
                        * self.kappa_seed
                    )
                    self.particulator.attributes["water mass"].data[npotseed_arr] += (
                        gamma_fac
                        * self.particulator.attributes["water mass"].data[potseed]
                        / len(npotseed_arr)
                    )
                    self.particulator.attributes["water mass"].data[potseed] = (
                        self.particulator.attributes["water mass"].data[potseed]
                        / len(npotseed_arr)
                    )
                except:
                    # ramdomly select a SD candidate to be the potential seed; tolerance increased to the seed radius
                    potseed_arr = np.where(
                        np.abs(
                            self.particulator.attributes["radius"].data - self.r_seed
                        )
                        < self.r_seed
                    )[0]
                    potseed_arr = np.where(
                        (
                            self.particulator.attributes["position in cell"].data[
                                0, potseed_arr
                            ]
                            > self.seed_z_part[0]
                        )
                        & (
                            self.particulator.attributes["position in cell"].data[
                                0, potseed_arr
                            ]
                            <= self.seed_z_part[1]
                        )
                    )[0]
                    potseed = np.random.choice(potseed_arr[potindx_arr], 1)[0]

                    npotseed_arr = np.where(ncell_arr == ncell_arr[potseed])[0]
                    npotseed_arr = npotseed_arr[npotseed_arr != potseed]

                    gamma_fac = (
                        self.particulator.attributes["multiplicity"].data[potseed]
                        / self.particulator.attributes["multiplicity"].data[
                            npotseed_arr
                        ]
                    ).astype(int)
                    gamma_fac[gamma_fac == 0] = 1
                    self.particulator.attributes["multiplicity"].data[
                        potseed
                    ] = self.m_param
                    self.particulator.attributes["kappa times dry volume"].data[
                        potseed
                    ] = (
                        self.particulator.attributes["dry volume"].data[potseed]
                        * self.kappa_seed
                    )
                    self.particulator.attributes["water mass"].data[npotseed_arr] += (
                        gamma_fac
                        * self.particulator.attributes["water mass"].data[potseed]
                        / len(npotseed_arr)
                    )
                    self.particulator.attributes["water mass"].data[potseed] = (
                        self.particulator.attributes["water mass"].data[potseed]
                        / len(npotseed_arr)
                    )

            self.mpdata.update_advector_field()
            if "Displacement" in self.particulator.dynamics:
                self.particulator.dynamics["Displacement"].upload_courant_field(
                    (self.mpdata.advector / self.g_factor_vec,)
                )
            self.particulator.run(steps=1)

            self.save(i + 1)

        Outputs = namedtuple("Outputs", "products attributes")
        output_results = Outputs(self.output_products, self.output_attributes)
        return output_results
