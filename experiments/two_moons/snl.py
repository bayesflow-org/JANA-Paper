import sbibm
import torch
from sbi.inference import SNLE_A, prepare_for_sbi, simulate_for_sbi
from sbi.utils import likelihood_nn

import numpy as np
import time
from functools import partial

import two_moons_wiqvist_torch
from utils import sbi_network_setup
from config import torch_device


torch.set_default_device(torch_device)

x_o = torch.tensor([0, 0], device=torch_device)
x_o = x_o.flatten()


def two_moons_snl(
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

        density_estimator = inference.append_simulations(theta, x).train(
            max_num_epochs=50, learning_rate=learning_rate
        )
        posterior = inference.build_posterior(density_estimator).set_default_x(x_o)
        posteriors.append(posterior)
        proposal = posterior

    end = time.time()
    run_time = end - start

    # Posterior inference
    start = time.time()
    for i in range(num_rounds):
        theta_trained = posteriors[i].sample((num_posterior_samples,), x=x_o)
        theta_trained = theta_trained.reshape((num_posterior_samples, 2))

        np.savetxt(
            f"output/{dir_prefix}snl_post_round{i+1}_seed{seed}.csv",
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
            f"output/{dir_prefix}snl_post_pred_round{i+1}_seed{seed}.csv",
            x_trained.detach().cpu().numpy(),
            delimiter=",",
        )

    end = time.time()
    run_time_posterior_predictive_inference = (end - start) / num_rounds

    with open(f"output/{dir_prefix}snl_metadata_seed{seed}.txt", "w") as f:
        f.write(f"Training time: {run_time:.4f}\n")
        f.write(
            f"Inference time (posterior) avg per round: {run_time_posterior_inference:.4f}\n"
        )
        f.write(
            f"Inference time (posterior predictive) avg per round: {run_time_posterior_predictive_inference:.4f}\n"
        )


if __name__ == "__main__":
    import config

    for seed in [2]:  # SNLE is too slow for 1k posterior samples to run 10 repeats
        two_moons_snl(
            num_simulations_per_round=config.num_simulations_per_round,
            num_rounds=config.num_rounds,
            seed=seed,
            prior=two_moons_wiqvist_torch.prior_torch,
            simulator=two_moons_wiqvist_torch.simulator_torch,
            num_posterior_samples=config.num_posterior_samples,
            dir_prefix="simulator_1/",
        )

        two_moons_snl(
            num_simulations_per_round=config.num_simulations_per_round,
            num_rounds=config.num_rounds,
            seed=seed,
            prior=two_moons_wiqvist_torch.prior_torch,
            simulator=sbibm.get_task("two_moons").get_simulator(),
            num_posterior_samples=config.num_posterior_samples,
            dir_prefix="simulator_2/",
        )
