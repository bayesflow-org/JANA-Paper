import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU visibility

import numpy as np

from bayesflow.simulation import GenerativeModel, Prior, Simulator
from bayesflow.networks import InvertibleNetwork
from bayesflow.amortizers import (
    AmortizedPosterior,
    AmortizedLikelihood,
    AmortizedPosteriorLikelihood,
)
from bayesflow.trainers import Trainer
from bayesflow import benchmarks

import config
import two_moons_wiqvist_numpy


def prior_fun():
    return np.random.uniform(low=[-2.0, -2.0], high=[2.0, 2.0])


def configure_input(forward_dict):
    """Function to configure the simulated quantities (i.e., simulator outputs)
    into a neural network-friendly (BayesFlow) format.
    """

    # Prepare placeholder dict
    out_dict = {"posterior_inputs": {}, "likelihood_inputs": {}}

    # Add to keys for posterior inference
    out_dict["posterior_inputs"]["direct_conditions"] = forward_dict["sim_data"].astype(
        np.float32
    )
    out_dict["posterior_inputs"]["parameters"] = forward_dict["prior_draws"].astype(
        np.float32
    )

    # Add keys for likelihood inference
    out_dict["likelihood_inputs"]["observables"] = forward_dict["sim_data"].astype(
        np.float32
    )
    out_dict["likelihood_inputs"]["conditions"] = forward_dict["prior_draws"].astype(
        np.float32
    )

    return out_dict


def two_moons_jana(
    simulation_budget,
    seed,
    generator,
    batch_size=32,
    num_posterior_samples=1000,
    dir_prefix="",
):
    # seeds
    np.random.seed(seed)

    # Create amortizers
    inference_net = InvertibleNetwork(
        num_params=2,
        num_coupling_layers=4,
        coupling_design="spline",
        permutation="learnable",
    )
    likelihood_net = InvertibleNetwork(
        num_params=2,
        num_coupling_layers=5,
        coupling_design="interleaved",
        permutation="learnable",
    )

    amortizer = AmortizedPosteriorLikelihood(
        AmortizedPosterior(inference_net), AmortizedLikelihood(likelihood_net)
    )

    # Create trainer
    trainer = Trainer(
        amortizer=amortizer,
        default_lr=0.0005,
        generative_model=generator,
        configurator=configure_input,
        memory=False,
    )

    # Generate data for offline learning
    sims = trainer.generative_model(simulation_budget)

    start = time.time()
    _ = trainer.train_offline(
        sims, epochs=64, batch_size=batch_size, validation_sims=300
    )
    end = time.time()
    run_time = end - start

    # Evaluate and save
    obs_data = {"direct_conditions": np.array([[0.0, 0.0]], dtype=np.float32)}

    # Posterior Inference
    start = time.time()
    post_samples = amortizer.sample_parameters(obs_data, num_posterior_samples)
    end = time.time()
    run_time_posterior_inference = end - start
    np.savetxt(
        f"output/{dir_prefix}jana_post_sims{simulation_budget}_seed{seed}.csv",
        post_samples,
        delimiter=",",
    )

    # Posterior Predictive Inference
    start = time.time()
    post_samples = amortizer.sample_parameters(obs_data, num_posterior_samples)
    post_pred_samples = amortizer.sample_data(
        {"conditions": post_samples}, n_samples=1
    ).squeeze()
    end = time.time()
    run_time_posterior_predictive_inference = end - start

    np.savetxt(
        f"output/{dir_prefix}jana_post_pred_sims{simulation_budget}_seed{seed}.csv",
        post_pred_samples,
        delimiter=",",
    )

    if simulation_budget == 10000:
        with open(f"output/{dir_prefix}jana_metadata_seed{seed}.txt", "w") as f:
            f.write(f"Training time: {run_time:.4f}\n")
            f.write(
                f"Inference time (posterior) 10k: {run_time_posterior_inference:.4f}\n"
            )
            f.write(
                f"Inference time (posterior predictive) 10k: {run_time_posterior_predictive_inference:.4f}\n"
            )


if __name__ == "__main__":
    # Settings
    batch_sizes = [16, 32, 64]
    simulation_budgets = [2000, 6000, 10000]

    for seed in config.seeds:
        for batch_size, simulation_budget in zip(batch_sizes, simulation_budgets):
            # Simulator 1 (Wiqvist et al)
            prior_1 = Prior(prior_fun=prior_fun)
            simulator_1 = Simulator(
                batch_simulator_fun=two_moons_wiqvist_numpy.simulator_numpy_batched
            )
            generator_1 = GenerativeModel(prior=prior_1, simulator=simulator_1)

            two_moons_jana(
                simulation_budget=simulation_budget,
                seed=seed,
                generator=generator_1,
                batch_size=batch_size,
                num_posterior_samples=1000,
                dir_prefix="simulator_1/",
            )

            # Simulator 2 (Lueckmann et al)
            generator_2 = benchmarks.Benchmark(
                "two_moons", prior_kwargs={"lower_bound": -2, "upper_bound": 2}
            ).generative_model

            two_moons_jana(
                simulation_budget=simulation_budget,
                seed=seed,
                generator=generator_2,
                batch_size=batch_size,
                num_posterior_samples=1000,
                dir_prefix="simulator_2/",
            )
