import numpy as np
import feather
import os
import tensorflow as tf
import pickle

from bayesflow.amortizers import AmortizedLikelihood
from bayesflow.networks import InvertibleNetwork, InvariantNetwork
from bayesflow.trainers import Trainer
from bayesflow.mcmc import PyMCSurrogateLikelihood

import pymc as pm
import arviz as az
import aesara.tensor as at

trial_number = 100

### load network
param_names = ["v", "a", "t0", "w"]

# specify the network architecture
from tensorflow_probability import distributions as tfd

coupling_net_settings={
    't_args': {
        'dense_args': dict(units=32, kernel_initializer='glorot_uniform', activation='tanh'),
        'num_dense': 2,
        'spec_norm': False
    },
    's_args': {
        'dense_args': dict(units=32, kernel_initializer='glorot_uniform', activation='tanh'),
        'num_dense': 2,
        'spec_norm': False
    },
}
likelihood_net = InvertibleNetwork(num_params=2, num_coupling_layers=12, coupling_net_settings=coupling_net_settings)

# Robust model training and generalisation with Studentising flows: https://arxiv.org/pdf/2006.06599.pdf
df = 50
mvt = tfd.MultivariateStudentTLinearOperator(
    df=df,
    loc=np.zeros(2, dtype=np.float32),
    scale=tf.linalg.LinearOperatorDiag(np.ones(2, dtype=np.float32))
)

amortized_likelihood = AmortizedLikelihood(likelihood_net, latent_dist=mvt)

# load from checkpoint
trainer_likelihood = Trainer(
    amortizer=amortized_likelihood,
    checkpoint_path="checkpoints/likelihood/"
)


def configurator(forward_dict, min_trials=500, max_trials=500):
    """ Configures simulator outputs for use in BayesFlow training."""
    N = np.random.randint(min_trials, max_trials + 1)
    forward_dict['sim_data'] = forward_dict['sim_data'][:,:N,1]
    
    # We need two variables, but reaction times are only one
    # -> Create dummy variable
    dummy = np.random.default_rng().normal(0, 1, forward_dict['sim_data'].shape)
    forward_dict['sim_data'] = np.stack((forward_dict['sim_data'], dummy), axis=-1)

    input_dict = {}
    input_dict['conditions'] = forward_dict['prior_draws'].astype(np.float32)
    input_dict['observables'] = forward_dict['sim_data'].astype(np.float32)
    return input_dict


from utility_functions import feather_loader

# Load data
N = 50
val_data = feather_loader("assets/evaluation/sim_data/01.feather",
                          param_names, num_simulations=N)
# Call configurator to bring the data into the required form
val_dict = configurator(val_data, min_trials=trial_number, max_trials=trial_number)
# Change the dummy variable from random values to all zeros
# to allow for comparison
sim_data = val_dict['observables']
dummy = np.zeros((sim_data.shape[0], sim_data.shape[1], 1))
sim_data[:,:,1:] = dummy
# extract the data and prior samples
prior_samples = val_dict['conditions']


def sample(data, num_chains, num_draws):
    # specifying the likelihood (takes the data set and likelihood as input)
    logl = PyMCSurrogateLikelihood(
        amortized_likelihood,
        data
    )

    # use PyMC to sample from log-likelihood
    with pm.Model() as model:
        # specify priors
        v = pm.TruncatedNormal("v", mu=0, sigma=10, lower=-5, upper=5)
        a = pm.TruncatedNormal("a", mu=1, sigma=1, lower=0.5, upper=3.0)
        t0 = pm.TruncatedNormal("t0", mu=0.4, sigma=0.2, lower=0.2, upper=1.0)
        w = pm.TruncatedNormal("w", mu=0.5, sigma=0.1, lower=0.3, upper=0.7)

        # convert parameters to tensor vector
        theta = at.as_tensor_variable([v, a, t0, w])

        # use Potential to add surrogate likelihood
        pm.Potential("likelihood", logl(theta))

        # Sample: As the SurrogateLikelihood uses the GPU,
        # only one core can be currently used
        # idata = pm.sample(cores=1, chains=num_chains, target_accept=0.8, tune=2000, draws=num_draws)
        idata = pm.sample(cores=1, chains=num_chains, tune=500, draws=num_draws)
    return idata


fit_path = f"data/osfstorage/fits_pymc/{trial_number}/evaluation"
os.makedirs(fit_path, exist_ok=True)

num_chains = 4
num_draws = 500
samples_pymc = np.zeros((N, num_chains, num_draws, 4))

for i in range(N):
    print("Sampling", i)
    out_file = f"{fit_path}/{i:03}.pickle"
    data = sim_data[i,:trial_number]
    idata = sample(data, num_chains=num_chains, num_draws=num_draws)
    with open(out_file, 'wb') as f:
        pickle.dump(idata, f, pickle.HIGHEST_PROTOCOL)
    samples_pymc[i] = np.stack([
        idata['posterior']['v'].data,
        idata['posterior']['a'].data,
        idata['posterior']['t0'].data,
        idata['posterior']['w'].data,
    ], axis=-1)
    np.save(f"assets/samples_pymc.npy", samples_pymc)
