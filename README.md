[![Travis Build Status](https://img.shields.io/travis/atmos-cloud-sim-uj/PySDM/master.svg?logo=travis)](https://travis-ci.org/atmos-cloud-sim-uj/PySDM)
[![Github Actions Build Status](https://github.com/atmos-cloud-sim-uj/PySDM/workflows/Build%20Status/badge.svg?branch=master)](https://github.com/atmos-cloud-sim-uj/PySDM/actions)
[![Appveyor Build status](http://ci.appveyor.com/api/projects/status/github/atmos-cloud-sim-uj/PySDM?branch=master&svg=true)](https://ci.appveyor.com/project/slayoo/pysdm/branch/master)
[![Coverage Status](https://codecov.io/gh/atmos-cloud-sim-uj/PySDM/branch/master/graph/badge.svg)](https://codecov.io/github/atmos-cloud-sim-uj/PySDM?branch=master)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
[![OpenHub](https://www.openhub.net/p/atmos-cloud-sim-uj-PySDM/widgets/project_thin_badge?format=gif)](https://www.openhub.net/p/atmos-cloud-sim-uj-PySDM)

# PySDM
PySDM is a package for simulating the dynamics of population of particles 
  immersed in moist air using the particle-based (a.k.a. super-droplet) approach 
  to represent aerosol/cloud/rain microphysics.
The package core is a Pythonic high-performance implementation of the 
  Super-Droplet Method (SDM) Monte-Carlo algorithm for representing collisional growth 
  ([Shima et al. 2009](http://doi.org/10.1002/qj.441)), hence the name. 
PySDM has two alternative parallel number-crunching backends 
  available: multi-threaded CPU backend based on [Numba](http://numba.pydata.org/) 
  and GPU-resident backend built on top of [ThrustRTC](https://pypi.org/project/ThrustRTC/).

## Dependencies and installation

It is worth here to distinguish the dependencies of the PySDM core subpackage 
(named simply ``PySDM``) vs. ``PySDM_examples`` and ``PySDM_tests`` subpackages.

PySDM core subpackage dependencies are all available through [PyPI](https://pypi.org), 
  the key dependencies are [Numba](http://numba.pydata.org/) and [Numpy](https://numpy.org/).

The **Numba backend** named ``CPU`` is the default, and features multi-threaded parallelism for 
  multi-core CPUs. 
It uses the just-in-time compilation technique based on the LLVM infrastructure.

The **ThrustRTC** backend named ``GPU`` offers GPU-resident operation of PySDM
  leveraging the [SIMT](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads) 
  parallelisation model. 

The dependencies of PySDM examples and test subpackages are summarised in
  the [requirements.txt](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/requirements.txt) 
  file.
Noteworthy, one of the examples (``Arabas_et_al_2015_Figs_8_9``) uses [PyMPDATA](https://github.com/atmos-cloud-sim-uj/PyMPDATA),
  a concurrently developed sister project to PySDM.

## Demo notebooks reproducing results from literature:
#### 0D box-model coalescence-only examples (work with both CPU and GPU backends):
- [Shima et al. 2009](http://doi.org/10.1002/qj.441) Fig. 2 
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PySDM.git/master?filepath=PySDM_examples/Shima_et_al_2009_Fig_2/demo.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atmos-cloud-sim-uj/PySDM/blob/master/PySDM_examples/Shima_et_al_2009_Fig_2/demo.ipynb)    
  (Box model, coalescence only, test case employing Golovin analytical solution)
- [Berry 1967](https://doi.org/10.1175/1520-0469(1967)024<0688:CDGBC>2.0.CO;2) Figs. 5, 8 & 10 
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PySDM.git/master?filepath=PySDM_examples/Berry_1967_Figs_5_8_10/demo.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atmos-cloud-sim-uj/PySDM/blob/master/PySDM_examples/Berry_1967_Figs_5_8_10/demo.ipynb)    
  (Box model, coalescence only, test cases for realistic kernels)
#### 0D parcel-model condensation only examples (work with CPU backend only)
- [Arabas & Shima 2017](http://dx.doi.org/10.5194/npg-24-535-2017) Fig. 5
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PySDM.git/master?filepath=PySDM_examples/Arabas_and_Shima_2017_Fig_5/demo.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atmos-cloud-sim-uj/PySDM/blob/master/PySDM_examples/Arabas_and_Shima_2017_Fig_5/demo.ipynb)    
  (Adiabatic parcel, monodisperse size spectrum activation/deactivation test case)
- [Yang et al. 2018](https://doi.org/10.5194/acp-18-7313-2018) Fig. 2:
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PySDM.git/master?filepath=PySDM_examples/Yang_et_al_2018_Fig_2/demo.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atmos-cloud-sim-uj/PySDM/blob/master/PySDM_examples/Yang_et_al_2018_Fig_2/demo.ipynb)    
  (Adiabatic parcel, polydisperse size spectrum activation/deactivation test case)
#### 2D kinematic-flow warm-rain example (works with CPU backend only)
- [Arabas et al. 2015](https://doi.org/10.5194/gmd-8-1677-2015) Figs. 8 & 9:
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PySDM.git/master?filepath=PySDM_examples/Arabas_et_al_2015_Figs_8_9/demo.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atmos-cloud-sim-uj/PySDM/blob/master/PySDM_examples/Arabas_et_al_2015_Figs_8_9/demo.ipynb)       
  (2D prescribed-flow stratocumulus-mimicking aerosol collisional processing test case)
  
## Hello-world example

In order to depict the PySDM API with a practical example, the following
  listings provide a sample code roughly reproducing the 
  Figure 2 from [Shima et al. 2009 paper](http://doi.org/10.1002/qj.441).
It is a coalescence-only set-up in which the initial particle size 
  spectrum is exponential and is deterministically sampled to match
  the condition of each super-droplet having equal initial multiplicity:
```Python
from PySDM.physics import si
from PySDM.initialisation.spectral_sampling import ConstantMultiplicity
from PySDM.initialisation.spectra import Exponential

n_sd = 2**15
initial_spectrum = Exponential(norm_factor=8.39e12, scale=1.19e5 * si.um**3)
attributes = {}
attributes['volume'], attributes['n'] = ConstantMultiplicity(spectrum=initial_spectrum).sample(n_sd)
```

The key element of the PySDM interface is the [``Core``](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/simulation/core.py) 
  class which instances are used to manage the system state and control the simulation.
Instantiation of the ``Core`` class is handled by the ``Builder``
  as exemplified below:
```Python
from PySDM import Builder
from PySDM.environments import Box
from PySDM.dynamics import Coalescence
from PySDM.dynamics.coalescence.kernels import Golovin
from PySDM.backends import CPU
from PySDM.products.state import ParticlesVolumeSpectrum

builder = Builder(n_sd=n_sd, backend=CPU)
builder.set_environment(Box(dt=1 * si.s, dv=1e6 * si.m**3))
builder.add_dynamic(Coalescence(kernel=Golovin(b=1.5e3 / si.s)))
products = [ParticlesVolumeSpectrum()]
particles = builder.build(attributes, products)
```
The ``backend`` argument may be set to ``CPU`` or ``GPU``
  what translates to choosing the multi-threaded backend or the 
  GPU-resident computation mode, respectively.
The employed ``Box`` environment corresponds to a zero-dimensional framework
  (particle positions are not considered).
The vectors of particle multiplicities ``n`` and particle volumes ``v`` are
  used to initialise super-droplet attributes.
The ``Coalescence`` Monte-Carlo algorithm (Super Droplet Method) is registered as the only
  dynamic in the system (other available dynamics representing
  condensational growth and particle displacement).
Finally, the ``build()`` method is used to obtain an instance
  of ``Core`` which can then be used to control time-stepping and
  access simulation state.

The ``run(nt)`` method advances the simulation by ``nt`` timesteps.
In the listing below, its usage is interleaved with plotting logic
  which displays a histogram of particle mass distribution 
  at selected timesteps:
```Python
from PySDM.physics.constants import rho_w
from matplotlib import pyplot
import numpy as np

radius_bins_edges = np.logspace(np.log10(10 * si.um), np.log10(5e3 * si.um), num=32)

for step in [0, 1200, 2400, 3600]:
    particles.run(step - particles.n_steps)
    pyplot.step(x=radius_bins_edges[:-1] / si.um,
                y=particles.products['dv/dlnr'].get(radius_bins_edges) * rho_w / si.g,
                where='post', label=f"t = {step}s")

pyplot.xscale('log')
pyplot.xlabel('particle radius [µm]')
pyplot.ylabel("dm/dlnr [g/m$^3$/(unit dr/r)]")
pyplot.legend()
pyplot.savefig('readme.svg')
```
The resultant plot looks as follows:

![plot](https://raw.githubusercontent.com/atmos-cloud-sim-uj/PySDM/master/readme.svg)

## Package structure and API

- [backends](https://github.com/atmos-cloud-sim-uj/PySDM/tree/master/PySDM/backends):
    - [CPU=Numba](https://github.com/piotrbartman/PySDM/tree/master/PySDM/backends/numba): 
      multi-threaded CPU backend using LLVM-powered just-in-time compilation
    - [GPU=ThrustRTC](https://github.com/piotrbartman/PySDM/tree/master/PySDM/backends/thrustRTC): 
      GPU-resident backend using NVRTC runtime compilation library for CUDA 
- [initialisation](https://github.com/atmos-cloud-sim-uj/PySDM/tree/master/PySDM/initialisation):
    - [multiplicities](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/initialisation/multiplicities.py): 
      integer-valued discretisation with sanity checks for errors due to type casting 
    - [r_wet_init](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/initialisation/r_wet_init.py):
      kappa-Keohler-based equilibrium in unsaturated conditions (RH=1 used in root-finding above saturation)
    - [spatial_sampling](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/initialisation/spatial_sampling.py): 
      pseudorandom sampling using NumPy's default RNG
    - [spectra](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/initialisation/spectra.py):
        Exponential and Lognormal classes
    - [spectral_sampling](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/initialisation/spectral_sampling.py):
        linear, logarithmic and constant_multiplicity classes
- [physics](https://github.com/atmos-cloud-sim-uj/PySDM/tree/master/PySDM/physics):
    - [constants](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/physics/constants.py): 
      physical constants partly imported from [SciPy](https://www.scipy.org/) and [mendeleev](https://pypi.org/project/mendeleev/) packages
    - [dimensional_analysis](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/physics/dimensional_analysis.py): 
      tool for enabling dimensional analysis of the code for unit tests (based on [pint](https://pint.readthedocs.io/))
    - [formulae](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/physics/formulae.py): 
      physical formulae partly imported from the Numba backend (e.g., for initialisation)
- [environments](https://github.com/atmos-cloud-sim-uj/PySDM/tree/master/PySDM/environments):
    - [Box](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/environments/box.py): 
      bare zero-dimensional framework 
    - [Parcel](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/environments/parcel.py): 
      zero-dimensional adiabatic parcel framework
    - [Kinematic2D](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/environments/kinematic_2d.py): 
      two-dimensional prescribed-flow-coupled framework with Eulerian advection handled by [PyMPDATA](http://github.com/atmos-cloud-sim-uj/PyMPDATA/)
- [dynamics](https://github.com/atmos-cloud-sim-uj/PySDM/tree/master/PySDM/dynamics):
    - [Coalescence](https://github.com/atmos-cloud-sim-uj/PySDM/tree/master/PySDM/dynamics/coalescence)
        - [coalescence.kernels (selected)](https://github.com/atmos-cloud-sim-uj/PySDM/tree/master/PySDM/dynamics/coalescence/kernels)
            - [Golovin](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/dynamics/coalescence/kernels/golovin.py)
            - [Geometric](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/dynamics/coalescence/kernels/geometric.py)
            - [Hydrodynamic](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/dynamics/coalescence/kernels/hydrodynamic.py)
            - ...
    - [Condensation](https://github.com/atmos-cloud-sim-uj/PySDM/tree/master/PySDM/dynamics/condensation.py)
        - solvers (working in arbitrary spectral coordinate specified through external class, defaults to logarithm of volume): 
            - [default](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/backends/numba/impl/condensation_methods.py):
              bespoke solver with implicit-in-particle-size integration and adaptive timestepping (Numba only as of now, soon on all backends)
            - [BDF](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM_tests/smoke_tests/utils/bdf.py): 
              black-box SciPy-based solver for benchmarking (Numba backend only)
    - [Displacement](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/dynamics/displacement.py):
      includes advection with the flow & sedimentation)
    - [EulerianAdvection](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/dynamics/eulerian_advection)
- Attributes (selected):
    - [cell](https://github.com/atmos-cloud-sim-uj/PySDM/tree/master/PySDM/attributes/cell):
        - [position_in_cell](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/attributes/cell/position_in_cell.py)
        - [cell_id](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/attributes/cell/cell_id.py)
        - ...
    - [droplet](https://github.com/atmos-cloud-sim-uj/PySDM/tree/master/PySDM/attributes/droplet):
        - [volume](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/attributes/droplet/volume.py)
        - [multiplicities](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/attributes/droplet/multiplicities.py)
        - [critical_radius](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/attributes/droplet/critical_radius.py)
        - ...
- Products (selected):
    - [SuperDropletCount](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/products/state/super_droplet_count.py)
    - [ParticlesVolumeSpectrum](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/products/state/particles_volume_spectrum.py)
    - [CondensationTimestep](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/products/dynamics/condensation/condensation_timestep.py)    
    - ...

## Credits:

Development of PySDM is supported by the EU through a grant of the Foundation for Polish Science (POIR.04.04.00-00-5E1C/18).

copyright: Jagiellonian University   
licence: GPL v3   

## SDM patents (some expired, some withdrawn):
- https://patents.google.com/patent/US7756693B2
- https://patents.google.com/patent/EP1847939A3
- https://patents.google.com/patent/JP4742387B2
- https://patents.google.com/patent/CN101059821B

## Other open-source SDM implementations:
- SCALE-SDM (Fortran):    
  https://github.com/Shima-Lab/SCALE-SDM_BOMEX_Sato2018/blob/master/contrib/SDM/sdm_coalescence.f90
- Pencil Code (Fortran):    
  https://github.com/pencil-code/pencil-code/blob/master/src/particles_coagulation.f90
- PALM LES (Fortran):    
  https://palm.muk.uni-hannover.de/trac/browser/palm/trunk/SOURCE/lagrangian_particle_model_mod.f90
- libcloudph++ (C++):    
  https://github.com/igfuw/libcloudphxx/blob/master/src/impl/particles_impl_coal.ipp
- LCM1D (Python)    
  https://github.com/SimonUnterstrasser/ColumnModel
