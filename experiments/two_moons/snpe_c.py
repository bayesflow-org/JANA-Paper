import sbi.inference
import sbibm
import torch
from sbi.inference import prepare_for_sbi, simulate_for_sbi, SNPE

import numpy as np
import time

import two_moons_wiqvist_torch
from config import torch_device

torch.set_default_device(torch_device)

x_o = torch.tensor([0, 0], device=torch_device)
x_o = x_o.flatten()


def two_moons_snpe_c(
    num_simulations_per_round,
    num_rounds,
    seed,
    prior,
    simulator,
    num_posterior_samples=1000,
    dir_prefix="",
):
    simulator, prior = prepare_for_sbi(simulator, prior)

    inference = SNPE(prior=prior, density_estimator="nsf", device=torch_device)

    learning_rate = 0.0005  # default value

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

        density_estimator = inference.append_simulations(
            theta, x, proposal=proposal
        ).train(max_num_epochs=50, learning_rate=learning_rate)
        posterior = inference.build_posterior(
            density_estimator, rejection_sampling_parameters={"enable_transform": False}
        )
        assert isinstance(posterior, sbi.inference.DirectPosterior)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(x_o)

    end = time.time()
    run_time = end - start

    # run and save inference for each iteration
    start = time.time()

    for i in range(num_rounds):
        theta_trained = posteriors[i].sample((num_posterior_samples,), x=x_o)
        theta_trained = theta_trained.reshape((num_posterior_samples, 2))

        np.savetxt(
            f"output/{dir_prefix}snpec_post_round{i+1}_seed{seed}.csv",
            theta_trained.detach().cpu().numpy(),
            delimiter=",",
        )

    end = time.time()
    run_time_posterior_inference = (end - start) / num_rounds

    with open(f"output/{dir_prefix}snpec_metadata_seed{seed}.txt", "w") as f:
        f.write(f"Training time: {run_time:.4f}\n")
        f.write(
            f"Inference time (posterior) avg per round: {run_time_posterior_inference:.4f}\n"
        )


if __name__ == "__main__":
    import config

    for seed in config.seeds:
        two_moons_snpe_c(
            num_simulations_per_round=config.num_simulations_per_round,
            num_rounds=config.num_rounds,
            seed=seed,
            prior=two_moons_wiqvist_torch.prior_torch,
            simulator=two_moons_wiqvist_torch.simulator_torch,
            num_posterior_samples=config.num_posterior_samples,
            dir_prefix="simulator_1/",
        )

        two_moons_snpe_c(
            num_simulations_per_round=config.num_simulations_per_round,
            num_rounds=config.num_rounds,
            seed=seed,
            prior=two_moons_wiqvist_torch.prior_torch,
            simulator=sbibm.get_task("two_moons").get_simulator(),
            num_posterior_samples=config.num_posterior_samples,
            dir_prefix="simulator_2/",
        )
