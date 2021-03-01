from PySDM_examples.Shipway_and_Hill_2012 import Simulation, Settings
from PySDM.physics import si
import numpy as np


def test_few_steps(plot=True):
    # Arrange
    settings = Settings(n_sd_per_gridbox=25, dt=60*si.s, dz=100*si.m)
    simulation = Simulation(settings)

    # Act
    output = simulation.run(nt=100)

    # Plot
    def profile(var):
        return np.mean(output[var][:, -20:], axis=1)

    if plot:
        from matplotlib import pyplot
        for var in ('RH_env', 'S_max', 'T_env', 'qv_env', 'p_env', 'ql', 'ripening_rate', 'activating_rate', 'deactivating_rate'):
            pyplot.plot(profile(var), output['z'], linestyle='--', marker='o')
            pyplot.ylabel('Z [m]')
            pyplot.xlabel(var + ' [' + simulation.core.products[var].unit + ']')
            pyplot.grid()
            pyplot.show()

    # Assert
    assert min(profile('ql')) == 0
    assert .2 < max(profile('ql')) < .5
    assert max(profile('ripening_rate')) > 0
    assert max(profile('activating_rate')) == 0
    assert max(profile('deactivating_rate')) > 0
