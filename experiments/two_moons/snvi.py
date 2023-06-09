import os

import sbibm
import torch
import numpy as np
import time

import sbi
from sbi.inference import (
    SNLE_A,
    ImportanceSamplingPosterior,
    likelihood_estimator_based_potential,
)
from sbi.inference import prepare_for_sbi, simulate_for_sbi
from sbi.utils import likelihood_nn

import two_moons_wiqvist_torch
from config import torch_device

torch.set_default_device(torch_device)

# Full parameter set used in the paper
# https://github.com/mackelab/snvi_repo/blob/main/pck3/config/experiments/SNL_two_moons_vi_fKL_correction.yaml
# https://github.com/mackelab/snvi_repo/blob/main/pck3/config/task/two_moons.yaml
# https://github.com/mackelab/snvi_repo/blob/main/pck3/config/method/snlvi.yaml
vi_parameters = {
    "flow": "affine_autoregressive",
    "num_flows": 5,
    "num_components": 1,
    "loss": "forward_kl",
    "n_particles": 256,
    "K": 8,
    "alpha": 0.5,
    "proposal": "q",
    "learning_rate": 1e-3,
    "gamma": 0.9999,
    "max_num_iters": 1000,
    "min_num_iters": 100,
    "check_for_convergence": True,
    "show_progress_bars": True,
    "reduce_variance": False,
    "sampling_method": "ir",
    "bound": 15,
    "count_bins": 15,
}

likelihood_params = {
    "num_rounds": 10,
    "neural_net": "maf",
    "hidden_features": 50,
    "simulation_batch_size": 1000,
    "training_batch_size": 1000,
    "automatic_transforms_enabled": False,
    "z_score_x": True,
    "z_score_theta": True,
    "simulation_filter": "identity",
    "samples_to_accept": None,
    "num_simulations_list": None,
}

x_o = torch.tensor([0, 0], device=torch_device)
x_o = x_o.flatten()


def two_moons_snvi(
    num_simulations_per_round,
    num_rounds,
    seed,
    prior,
    simulator,
    num_posterior_samples=1000,
    dir_prefix="",
):
    simulator, prior = prepare_for_sbi(simulator, prior)

    density_estimator_fun = likelihood_nn(
        model="maf",
        hidden_features=50,
        z_score_x="independent",
        z_score_theta="independent",
    )

    inference = SNLE_A(
        prior=prior,
        density_estimator=density_estimator_fun,
        device=torch_device,
        show_progress_bars=False,
    )

    learning_rate = 0.0005

    torch.manual_seed(seed)
    np.random.seed(seed)

    start = time.time()

    posteriors = []
    proposal = prior

    for i in range(num_rounds):
        theta, x = simulate_for_sbi(
            simulator=simulator,
            proposal=proposal,
            num_simulations=num_simulations_per_round,
        )

        likelihood_estimator = inference.append_simulations(theta, x).train(
            learning_rate=learning_rate,
            training_batch_size=1000,
        )
        potential, transform = likelihood_estimator_based_potential(
            likelihood_estimator, prior, x_o
        )

        if i == 0:
            vi_posterior = inference.build_posterior(
                sample_with="vi", vi_method="fKL", vi_parameters={"q": "nsf"}
            )
        else:
            vi_posterior = inference.build_posterior(
                sample_with="vi", vi_method="fKL", vi_parameters={"q": vi_posterior}
            )

        vi_posterior = vi_posterior.set_default_x(x_o)
        vi_posterior.train(
            learning_rate=1e-3,
            gamma=0.9999,
            max_num_iters=1000,
            min_num_iters=100,
        )
        proposal = ImportanceSamplingPosterior(
            potential, vi_posterior, transform, oversampling_factor=8
        )
        posteriors.append(proposal)

    end = time.time()
    run_time = end - start

    # Posterior inference
    start = time.time()
    for i in range(num_rounds):
        theta_trained = posteriors[i].sample((num_posterior_samples,), x=x_o)
        theta_trained = theta_trained.reshape((num_posterior_samples, 2))

        np.savetxt(
            f"output/{dir_prefix}snvi_post_round{i+1}_seed{seed}.csv",
            theta_trained.detach().cpu().numpy(),
            delimiter=",",
        )

    end = time.time()
    run_time_posterior_inference = (end - start) / num_rounds

    # Posterior Predictive inference
    start = time.time()
    for i in range(num_rounds):
        theta_trained = posteriors[i].sample((num_posterior_samples,), x=x_o)
        theta_trained = theta_trained.reshape((num_posterior_samples, 2))

        x_trained = inference._neural_net.sample(1, context=theta_trained)
        x_trained = x_trained.reshape((num_posterior_samples, 2))

        np.savetxt(
            f"output/{dir_prefix}snvi_post_pred_round{i+1}_seed{seed}.csv",
            x_trained.detach().cpu().numpy(),
            delimiter=",",
        )

    end = time.time()
    run_time_posterior_predictive_inference = (end - start) / num_rounds

    with open(f"output/{dir_prefix}snvi_metadata_seed{seed}.txt", "w") as f:
        f.write(f"Training time: {run_time:.4f}\n")
        f.write(
            f"Inference time (posterior) avg per round: {run_time_posterior_inference:.4f}\n"
        )
        f.write(
            f"Inference time (posterior predictive) avg per round: {run_time_posterior_predictive_inference:.4f}\n"
        )


if __name__ == "__main__":
    import config

    for seed in config.seeds:
        two_moons_snvi(
            num_simulations_per_round=config.num_simulations_per_round,
            num_rounds=config.num_rounds,
            seed=seed,
            prior=two_moons_wiqvist_torch.prior_torch,
            simulator=two_moons_wiqvist_torch.simulator_torch,
            num_posterior_samples=config.num_posterior_samples,
            dir_prefix="simulator_1/",
        )

        two_moons_snvi(
            num_simulations_per_round=config.num_simulations_per_round,
            num_rounds=config.num_rounds,
            seed=seed,
            prior=two_moons_wiqvist_torch.prior_torch,
            simulator=sbibm.get_task("two_moons").get_simulator(),
            num_posterior_samples=config.num_posterior_samples,
            dir_prefix="simulator_2/",
        )
