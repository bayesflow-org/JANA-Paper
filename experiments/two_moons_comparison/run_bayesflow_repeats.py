import sys
import os
sys.path.append(os.path.abspath(os.path.join('../../../BayesFlow')))

import tensorflow_probability as tfp
import numpy as np
import pickle

from bayesflow.simulation import GenerativeModel, Prior, Simulator
from bayesflow.networks import InvertibleNetwork
from bayesflow.amortizers import AmortizedPosterior, AmortizedLikelihood, AmortizedPosteriorLikelihood
from bayesflow.trainers import Trainer

from simulation import TwoMoons

tfd = tfp.distributions


SETTINGS_POST = {
    't_args': {
        'dense_args': dict(units=256, activation='tanh'), 'n_dense': 2, 'dropout': True, 'dropout_prob': 0.05
    },
    's_args': {
        'dense_args': dict(units=256, activation='tanh'), 'n_dense': 2, 'dropout': True, 'dropout_prob': 0.05
    }
}

SETTINGS_LIK = {
    't_args': {
        'dense_args': dict(units=64, activation='tanh'), 'n_dense': 2, 'dropout': True, 'dropout_prob': 0.05
    },
    's_args': {
        'dense_args': dict(units=64, activation='tanh'), 'n_dense': 2, 'dropout': True, 'dropout_prob': 0.05
    }
}


def prior_fun():
    return np.random.uniform(low=[-2., -2.], high=[2., 2.])


def configure_input(forward_dict):
    """Function to configure the simulated quantities (i.e., simulator outputs)
    into a neural network-friendly (BayesFlow) format.
    """
    
    # Prepare placeholder dict
    out_dict = {
        'posterior_inputs': {},
        'likelihood_inputs': {}
    }
    
    # Add to keys for posterior inference
    out_dict['posterior_inputs']['direct_conditions'] = forward_dict['sim_data'].astype(np.float32)
    out_dict['posterior_inputs']['parameters'] = forward_dict['prior_draws'].astype(np.float32)
    
    # Add keys for likelihood inference
    out_dict['likelihood_inputs']['observables'] = forward_dict['sim_data'].astype(np.float32)
    out_dict['likelihood_inputs']['conditions'] = forward_dict['prior_draws'].astype(np.float32)
    
    return out_dict


def create_and_train(idx, num_sims, batch_size=128, num_epochs=128, num_val=300, learning_rate=0.001):
    
    ### Create amortizers
    amortizer = AmortizedPosteriorLikelihood(
        AmortizedPosterior(
            InvertibleNetwork(num_params=2, num_coupling_layers=8, coupling_net_settings=SETTINGS_POST), latent_dist=mixture),
        AmortizedLikelihood(
            InvertibleNetwork(num_params=2, num_coupling_layers=6, coupling_net_settings=SETTINGS_LIK))
    )
    
    ### Create trainer
    trainer = Trainer(
        amortizer=amortizer,
        default_lr=learning_rate,
        generative_model=generator,
        configurator=configure_input,
        checkpoint_path=f'checkpoints/two_moons_{idx}_{num_sims}',
        memory=False   
    )
    
    ### Generate data for offline learning
    sims = trainer.generative_model(num_sims)
    
    ### Train with early stopping
    h = trainer.train_offline(
        sims, epochs=num_epochs, batch_size=batch_size, early_stopping=True, validation_sims=num_val)
    return h, amortizer


def evaluate(amortizer, num_post_samples=1000):

    # Prepare output dictionary and data
    samples_dict = {}
    obs_data = {'direct_conditions': np.array([[0.,0.]], dtype=np.float32)}
    post_samples = amortizer.sample_parameters(obs_data, num_post_samples)
    lik_samples = amortizer.sample_data({'conditions': post_samples}, n_samples=1).squeeze()

    samples_dict['post_samples'] = post_samples
    samples_dict['lik_samples'] = lik_samples
    return samples_dict

if __name__ == "__main__":

    ### Latent distro 
    mixture = tfp.distributions.Mixture(
        cat=tfd.Categorical(probs=[0.5, 0.5]),
        components=[
            tfd.MultivariateNormalDiag(loc=[-2, -2], scale_diag=[1., 1.]),
            tfd.MultivariateNormalDiag(loc=[2, 2], scale_diag=[1., 1.]),
        ]
    )

    ### Generative model
    prior = Prior(prior_fun = prior_fun, param_names=[r'$\theta_1$', r'$\theta_2$'])
    tm = TwoMoons(mean_radius=1.0, sd_radius=0.1, baseoffset=1.0)
    simulator = Simulator(batch_simulator_fun=tm.model_sim_numpy)
    generator = GenerativeModel(prior=prior, simulator=simulator)

    # Number of repeats
    num_repeats = 10
    for num_sims in [2000, 6000, 10000]:
        for repetition_id in range(1, num_repeats+1):
            print(f'Starting run: simulations: {num_sims}, instance: {repetition_id}')

            # Complete training, evaluate, and store
            history, amortizer = create_and_train(repetition_id, num_sims)
            samples_dict = evaluate(amortizer)

            with open(f'./assests/bf_samples_{num_sims}_{repetition_id}.pkl', 'wb') as file:
                pickle.dump(samples_dict, file)