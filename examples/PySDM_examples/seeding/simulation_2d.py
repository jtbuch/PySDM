import numpy as np
from PySDM_examples.utils.kinematic_2d.make_default_product_collection import (
    make_default_product_collection,
)
from PySDM_examples.utils.kinematic_2d import MPDATA_2D
from PySDM_examples.utils import DummyController

from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import (
    AmbientThermodynamics,
    Coalescence,
    Collision,
    Condensation,
    Displacement,
    EulerianAdvection,
    Freezing,
    Seeding,
)
from PySDM.impl.mesh import Mesh
from PySDM.environments import Kinematic2D
from PySDM.initialisation import spectra
from PySDM.initialisation.sampling import spatial_sampling, spectral_sampling
from PySDM.initialisation.equilibrate_wet_radii import equilibrate_wet_radii
from PySDM.physics import si


class Simulation:
    def __init__(self, settings, storage, SpinUp, backend_class=CPU):
        self.settings = settings
        self.storage = storage
        self.particulator = None
        self.backend_class = backend_class
        self.SpinUp = SpinUp

    @property
    def products(self):
        return self.particulator.products

    def reinit(self, products=None):
        formulae = self.settings.formulae
        backend = self.backend_class(formulae=formulae)
        self.mesh = Mesh(grid=self.settings.grid, size=self.settings.size)
        environment = Kinematic2D(
            dt=self.settings.dt,
            grid=self.settings.grid,
            size=self.settings.size,
            rhod_of=self.settings.rhod_of_zZ,
            mixed_phase=self.settings.processes["freezing"],
        )
        builder = Builder(
            n_sd=self.settings.n_sd + self.settings.n_sd_seeding,
            backend=backend,
            environment=environment,
        )

        if products is not None:
            products = list(products)
        else:
            products = make_default_product_collection(self.settings)

        if self.settings.processes["fluid advection"]:
            builder.add_dynamic(AmbientThermodynamics())
        if self.settings.processes["condensation"]:
            kwargs = {}
            if not self.settings.condensation_adaptive:
                kwargs["substeps"] = (self.settings.condensation_substeps,)
            condensation = Condensation(
                rtol_x=self.settings.condensation_rtol_x,
                rtol_thd=self.settings.condensation_rtol_thd,
                adaptive=self.settings.condensation_adaptive,
                dt_cond_range=self.settings.condensation_dt_cond_range,
                schedule=self.settings.condensation_schedule,
                **kwargs,
            )
            builder.add_dynamic(condensation)
        if self.settings.processes["fluid advection"]:
            initial_profiles = {
                "th": self.settings.initial_dry_potential_temperature_profile,
                "water_vapour_mixing_ratio": self.settings.initial_vapour_mixing_ratio_profile,
            }
            advectees = dict(
                (
                    key,
                    np.repeat(
                        profile.reshape(1, -1),
                        builder.particulator.environment.mesh.grid[0],
                        axis=0,
                    ),
                )
                for key, profile in initial_profiles.items()
            )
            solver = MPDATA_2D(
                advectees=advectees,
                stream_function=self.settings.stream_function,
                rhod_of_zZ=self.settings.rhod_of_zZ,
                dt=self.settings.dt,
                grid=self.settings.grid,
                size=self.settings.size,
                n_iters=self.settings.mpdata_iters,
                infinite_gauge=self.settings.mpdata_iga,
                nonoscillatory=self.settings.mpdata_fct,
                third_order_terms=self.settings.mpdata_tot,
            )
            builder.add_dynamic(EulerianAdvection(solver))
        if self.settings.processes["particle advection"]:
            builder.add_dynamic(
                Displacement(
                    enable_sedimentation=self.settings.processes["sedimentation"],
                    adaptive=self.settings.displacement_adaptive,
                    rtol=self.settings.displacement_rtol,
                )
            )
        if (
            self.settings.processes["coalescence"]
            and self.settings.processes["breakup"]
        ):
            builder.add_dynamic(
                Collision(
                    collision_kernel=self.settings.kernel,
                    enable_breakup=self.settings.processes["breakup"],
                    coalescence_efficiency=self.settings.coalescence_efficiency,
                    breakup_efficiency=self.settings.breakup_efficiency,
                    fragmentation_function=self.settings.breakup_fragmentation,
                    adaptive=self.settings.coalescence_adaptive,
                    dt_coal_range=self.settings.coalescence_dt_coal_range,
                    substeps=self.settings.coalescence_substeps,
                    optimized_random=self.settings.coalescence_optimized_random,
                )
            )
        elif (
            self.settings.processes["coalescence"]
            and not self.settings.processes["breakup"]
        ):
            builder.add_dynamic(
                Coalescence(
                    collision_kernel=self.settings.kernel,
                    adaptive=self.settings.coalescence_adaptive,
                    dt_coal_range=self.settings.coalescence_dt_coal_range,
                    substeps=self.settings.coalescence_substeps,
                    optimized_random=self.settings.coalescence_optimized_random,
                )
            )
        assert not (
            self.settings.processes["breakup"]
            and not self.settings.processes["coalescence"]
        )
        if self.settings.processes["freezing"]:
            builder.add_dynamic(
                Freezing(
                    singular=self.settings.freezing_singular,
                    thaw=self.settings.freezing_thaw,
                )
            )

        attributes = builder.particulator.environment.init_attributes(
            spatial_discretisation=spatial_sampling.Pseudorandom(),
            dry_radius_spectrum=self.settings.spectrum_per_mass_of_dry_air,
            kappa=self.settings.kappa,
            n_sd=self.settings.n_sd
            // (2 if self.settings.freezing_inp_frac != 1 else 1),
        )

        r_dry, n_in_dv = spectral_sampling.ConstantMultiplicity(
            spectra.Lognormal(
                norm_factor=(
                    self.settings.seed_particles_per_volume_STP
                    / self.settings.formulae.constants.rho_STP
                ),
                m_mode=self.settings.seed_radius,
                s_geom=1.4,
            )
        ).sample(
            n_sd=self.settings.n_sd_seeding
        )  # TODO #1387: does not have to be the same?
        v_dry = self.settings.formulae.trivia.volume(radius=r_dry)
        self.seeded_particle_extensive_attributes = {
            "signed water mass": np.array(
                [0.0001 * si.ng] * self.settings.n_sd_seeding
            ),
            "dry volume": v_dry,
            "kappa times dry volume": self.settings.seed_kappa
            * v_dry,  # include kappa argument for seeds
        }
        self.seeded_particle_multiplicity = n_in_dv * np.prod(np.array(self.mesh.size))

        positions = spatial_sampling.Pseudorandom().sample(
            backend=backend,
            grid=self.mesh.grid,
            n_sd=self.settings.n_sd_seeding,
        )
        cell_id, cell_origin, pos_cell = self.mesh.cellular_attributes(positions)
        self.seeded_particle_cell_id = cell_id
        self.seeded_particle_cell_origin = cell_origin
        self.seeded_particle_pos_cell = pos_cell

        r_wet = equilibrate_wet_radii(
            r_dry=self.settings.formulae.trivia.radius(volume=v_dry),
            environment=builder.particulator.environment,
            cell_id=cell_id,
            kappa_times_dry_volume=self.settings.seed_kappa
            * v_dry,  # include kappa argument for seeds
        )
        self.seeded_particle_volume = self.settings.formulae.trivia.volume(radius=r_wet)

        builder.add_dynamic(
            Seeding(
                super_droplet_injection_rate=self.settings.super_droplet_injection_rate,
                seeded_particle_multiplicity=self.seeded_particle_multiplicity,
                seeded_particle_extensive_attributes=self.seeded_particle_extensive_attributes,
                seeded_particle_cell_id=self.seeded_particle_cell_id,
                seeded_particle_cell_origin=self.seeded_particle_cell_origin,
                seeded_particle_pos_cell=self.seeded_particle_pos_cell,
                seeded_particle_volume=self.seeded_particle_volume,
            )
        )

        if self.settings.processes["freezing"]:
            attributes["signed water mass"] = attributes.pop("water mass")

            if self.settings.freezing_inp_spec is None:
                immersed_surface_area = formulae.trivia.sphere_surface(
                    diameter=2 * formulae.trivia.radius(volume=attributes["dry volume"])
                )
            else:
                immersed_surface_area = self.settings.freezing_inp_spec.percentiles(
                    np.random.random(attributes["dry volume"].size),  # TODO #599: seed
                )

            if self.settings.freezing_singular:
                attributes["freezing temperature"] = (
                    formulae.freezing_temperature_spectrum.invcdf(
                        np.random.random(immersed_surface_area.size),  # TODO #599: seed
                        immersed_surface_area,
                    )
                )
            else:
                attributes["immersed surface area"] = immersed_surface_area

            if self.settings.freezing_inp_frac != 1:
                assert self.settings.n_sd % 2 == 0
                assert 0 < self.settings.freezing_inp_frac < 1
                freezing_attribute = {
                    True: "freezing temperature",
                    False: "immersed surface area",
                }[self.settings.freezing_singular]
                for name, array in attributes.items():
                    if array.shape[-1] != self.settings.n_sd // 2:
                        raise AssertionError(f"attribute >>{name}<< has wrong size")
                    array = array.copy()
                    full_shape = list(array.shape)
                    orig = slice(None, full_shape[-1])
                    copy = slice(orig.stop, None)
                    full_shape[-1] *= 2
                    attributes[name] = np.empty(full_shape, dtype=array.dtype)
                    if name == freezing_attribute:
                        attributes[name][orig] = array
                        attributes[name][copy] = 0
                    elif name == "multiplicity":
                        attributes[name][orig] = array * self.settings.freezing_inp_frac
                        attributes[name][copy] = array * (
                            1 - self.settings.freezing_inp_frac
                        )
                    elif len(array.shape) > 1:  # particle positions
                        # TODO #599: seed
                        for dim, _ in enumerate(array.shape):
                            # only to make particles not shadow each other in visualisations
                            attributes[name][dim, orig] = array[dim, :]
                            attributes[name][dim, copy] = np.random.permutation(
                                array[dim, :]
                            )
                    else:
                        attributes[name][orig] = array
                        attributes[name][copy] = array

                non_zero_per_gridbox = np.count_nonzero(
                    attributes[freezing_attribute]
                ) / np.prod(self.settings.grid)
                assert non_zero_per_gridbox == self.settings.n_sd_per_gridbox / 2

        self.particulator = builder.build(
            attributes={
                k: np.pad(
                    array=v,
                    pad_width=(
                        ((0, 0), (0, self.settings.n_sd_seeding))
                        if k in ("position in cell", "cell origin")
                        else (0, self.settings.n_sd_seeding)
                    ),
                    mode="constant",
                    constant_values=np.nan if k == "multiplicity" else 0,
                )
                for k, v in attributes.items()
            },
            products=tuple(products),
        )

        if self.SpinUp is not None:
            self.SpinUp(self.particulator, self.settings.n_spin_up)
        if self.storage is not None:
            self.storage.init(self.settings)

    def run(self, controller=DummyController(), vtk_exporter=None):
        with controller:
            for step in self.settings.output_steps:
                if controller.panic:
                    break

                self.particulator.run(step - self.particulator.n_steps)

                self.store(step)

                if vtk_exporter is not None:
                    vtk_exporter.export_attributes(self.particulator)
                    vtk_exporter.export_products(self.particulator)

                controller.set_percent(step / self.settings.output_steps[-1])

    def store(self, step):
        for name, product in self.particulator.products.items():
            self.storage.save(product.get(), step, name)
