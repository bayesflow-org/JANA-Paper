import os
import feather
import numpy as np


def feather_loader(file_path, param_names, num_simulations = -1):
    """Loads data and corresponding parameter file. If the path to the
    simulated data is /path/to/sim_data/filename.feather, the parameters are
    assumed to be in /path/to/parameters/filename.feather.

    file_path: Path to the sim_data file
    param_names: Parameters to load
    num_simulations: number of data sets that are returned
    """
    
    file_path = os.path.normpath(file_path)
    file_path_sep = file_path.split(os.sep)
    data = feather.read_dataframe(file_path, columns = ["cnd", "resp", "rt"]).to_numpy()
    file_path_sep[-2] = "parameters"
    params = feather.read_dataframe(os.path.join(*file_path_sep), columns=param_names).to_numpy()
    
    # reformat data
    data[:,0] = data[:,0] - 1
    data[:,2] = data[:,2] * ((data[:,1] - 0.5) * 2)
    
    data = data[:,[0,2]]

    num_sim = params.shape[0]
    num_params = params.shape[1]
    num_trials_file = data.shape[0] // num_sim
    num_data_cols = data.shape[1]

    sim_data = np.reshape(data, (num_sim, num_trials_file, num_data_cols))
    if num_simulations < 0:
        num_simulations = num_sim

    return {
        'prior_draws': params[:num_simulations],
        'sim_data': sim_data[:num_simulations]
    }
