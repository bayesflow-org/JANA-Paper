import cmdstanpy
from cmdstanpy import CmdStanModel
import numpy as np
import feather
import os

# set path to cmdstan installation (adjust to local setup)
cmdstanpy.set_cmdstan_path("../../cmdstan")

stan_file = "assets/stan/avt0w.stan"
stan_model = CmdStanModel(stan_file=stan_file,
                          cpp_options={"STAN_THREADS": False})

trial_number = 100


def load_data(N, file_path):
    file_path = os.path.normpath(file_path)
    file_path_sep = file_path.split(os.sep)
    data = feather.read_dataframe(file_path, columns = ["cnd", "resp", "rt"]).to_numpy()
    file_path_sep[-2] = "parameters"
    params = feather.read_dataframe(os.path.join(*file_path_sep)).to_numpy()
    
    num_sim = params.shape[0]
    num_trials = data.shape[0] // num_sim
    num_data_cols = data.shape[1]
        
    first = 0
    last = N

    sim_data = np.reshape(data[(first * num_trials):(last * num_trials)], (N, num_trials, num_data_cols))
    return sim_data


N = 2000
sim_data = load_data(N, "assets/evaluation/sim_data/01.feather")
fit_path = f"data/osfstorage/fits_stan/{trial_number}/evaluation"

samples_stan = np.zeros((N, 2000, 4))

for i in range(N):
    print("Sampling", i)
    out_dir = f"{fit_path}/{i:04}"
    data = sim_data[i,:trial_number]

    data = {
        'N': data.shape[0],
        'rt': data[:,2],
        'dummy': np.zeros(data.shape[0]),
        'resp': [int(resp) for resp in data[:,1]]
    }
    fit = stan_model.sample(data=data, iter_warmup=250, iter_sampling=500,
                            parallel_chains=4, output_dir=out_dir,
                            threads_per_chain=1, chains=4)
    variables = fit.stan_variables()
    samples_stan[i] = np.stack([variables['v'], variables['a'], variables['t0'], variables['w']], axis=-1)
    np.save(f"data/osfstorage/fits_stan/samples_{trial_number}.npy", samples_stan)
