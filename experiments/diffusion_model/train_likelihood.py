import os
import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from bayesflow.amortizers import AmortizedLikelihood
from bayesflow.networks import InvertibleNetwork
from bayesflow.trainers import Trainer

from utility_functions import feather_loader


print("GPUs:", tf.config.experimental.list_physical_devices('GPU'))


param_names = ["v", "a", "t0", "w"]

### Specify the network architecture
# as we only have a small number of inputs and of targets, we reduce the number of units per
# coupling layer. The swish activation is used based on positive experiences in previous runs
coupling_settings={
    't_args': {
        'dense_args': dict(units=32, kernel_initializer='glorot_uniform', activation='tanh'),
        'num_dense': 2,
    },
    's_args': {
        'dense_args': dict(units=32, kernel_initializer='glorot_uniform', activation='tanh'),
        'num_dense': 2,
    },
}
# we only use 10 layers, using more layers leads to decreased speed but more accurate approximation
likelihood_net = InvertibleNetwork(num_params=2, num_coupling_layers=12,
                                   coupling_settings=coupling_settings)

# Based on Alexanderson & Henter (2020): Robust model training and generalisation with
# Studentising flows: https://arxiv.org/pdf/2006.06599.pdf
df = 50
mvt = tfd.MultivariateStudentTLinearOperator(
    df=df,
    loc=np.zeros(2, dtype=np.float32),
    scale=tf.linalg.LinearOperatorDiag(np.ones(2, dtype=np.float32))
)

amortized_likelihood = AmortizedLikelihood(likelihood_net, latent_dist=mvt)



### Load and prepare the training data
presimulation_path = "data/osfstorage/likelihood/training/sim_data/"
file_list = os.listdir(presimulation_path)

training_data = None
for i in range(0, len(file_list)):
    print(f"Loading data set {i+1:02}\r", end="")
    # if the training data is split into multiple files, combine them
    add_data = feather_loader(presimulation_path + '/' + file_list[i],
                              param_names=param_names)
    if training_data is None:
        training_data = add_data
    else:
        training_data['prior_draws'] = np.concatenate((training_data['prior_draws'], add_data['prior_draws']))
        training_data['sim_data'] = np.concatenate((training_data['sim_data'], add_data['sim_data']))


# We can use a configurator to add a dummy variable, which will be passed to the likelihood network.
# This is needed to make the coupling layers work with one-dimensional inputs.
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

### Run the training using CosineDecay
initial_learning_rate = 0.001

trainer = Trainer(
    amortizer=amortized_likelihood,
    configurator=configurator,
    default_lr=initial_learning_rate,
    checkpoint_path="checkpoints/likelihood"
)

decay_steps = 400000
num_sim = training_data['sim_data'].shape[0]
batch_size = 64
# calculate number of epochs needed to get desired number of iterations
epochs = decay_steps // (num_sim // batch_size)

history = trainer.train_offline(training_data, epochs=epochs, batch_size=batch_size)
