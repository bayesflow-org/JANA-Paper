import os
import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import InvertibleNetwork, DeepSet
from bayesflow.trainers import Trainer

from utility_functions import feather_loader


print("GPUs:", tf.config.experimental.list_physical_devices('GPU'))


param_names = ["v", "a", "t0", "w"]

### Specify the network architecture
summary_net = DeepSet()
# Based on Alexanderson & Henter (2020): Robust model training and generalisation with
# Studentising flows: https://arxiv.org/pdf/2006.06599.pdf
df = 50
mvt = tfd.MultivariateStudentTLinearOperator(
    df=df,
    loc=np.zeros(len(param_names), dtype=np.float32),
    scale=tf.linalg.LinearOperatorDiag(np.ones(len(param_names), dtype=np.float32))
)
inference_net = InvertibleNetwork(
    num_params=len(param_names),
    num_coupling_layers=5
)

amortized_posterior = AmortizedPosterior(inference_net, summary_net, latent_dist=mvt)


### Load and prepare the training data
presimulation_path = "data/osfstorage/posterior/training/sim_data/"
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


def configurator(forward_dict, min_trials=80, max_trials=120):
    """ Configures simulator outputs for use in BayesFlow training."""
    N = np.random.randint(min_trials, max_trials + 1)
    forward_dict['sim_data'] = forward_dict['sim_data'][:,:N,1:]

    input_dict = {}
    input_dict['parameters'] = forward_dict['prior_draws'].astype(np.float32)
    input_dict['summary_conditions'] = forward_dict['sim_data'].astype(np.float32)
    
    # supply trial number to posterior network
    batch_size = forward_dict['prior_draws'].shape[0]
    # number of trials (identical for the whole batch)
    N_ones = np.ones((batch_size, 1))
    # supply the logarithmized number of trials directly to the inference network 
    input_dict['direct_conditions'] = (N_ones * np.log(N)).astype(np.float32)
    return input_dict

### Run the training using CosineDecay
initial_learning_rate = 0.002

trainer = Trainer(
    amortizer=amortized_posterior,
    configurator=lambda x: configurator(x, 80, 120),
    default_lr=initial_learning_rate,
    checkpoint_path="checkpoints/posterior_80_120"
)

decay_steps = 200000
num_sim = training_data['sim_data'].shape[0]
batch_size = 64
epochs = decay_steps // (num_sim // batch_size)

history = trainer.train_offline(training_data, epochs=epochs, batch_size=batch_size)
