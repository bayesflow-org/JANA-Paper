import torch
import numpy as np
import time

from functools import partial

import sbibm
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi

import two_moons_wiqvist_torch
from utils import sbi_network_setup
from config import torch_device

torch.set_default_device(torch_device)

x_o = torch.tensor([0, 0], device=torch_device)
x_o = x_o.flatten()


def two_moons_npe_c(
    simulation_budget, seed, prior, simulator, num_posterior_samples=1000, dir_prefix=""
):
    simulator, prior = prepare_for_sbi(simulator, prior)

    inference = SNPE(prior=prior, density_estimator="nsf", device=torch_device)

    learning_rate = 0.0005  # default value

    torch.manual_seed(seed)
    np.random.seed(seed)

    start = time.time()

    theta, x = simulate_for_sbi(
        simulator=simulator, proposal=prior, num_simulations=simulation_budget
    )

    density_estimator = inference.append_simulations(theta, x).train(
        max_num_epochs=50, learning_rate=learning_rate
    )
    posterior = inference.build_posterior(density_estimator).set_default_x(x_o)

    end = time.time()
    run_time = end - start

    # run and save inference for each iteration
    start = time.time()

    theta_trained = posterior.sample((num_posterior_samples,), x=x_o)
    theta_trained = theta_trained.reshape((num_posterior_samples, 2))

    np.savetxt(
        f"output/{dir_prefix}npec_post_sims{simulation_budget}_seed{seed}.csv",
        theta_trained.detach().cpu().numpy(),
        delimiter=",",
    )

    end = time.time()
    run_time_posterior_inference = end - start

    if simulation_budget == 10000:
        with open(f"output/{dir_prefix}npec_metadata_seed{seed}.txt", "w") as f:
            f.write(f"Training time: {run_time:.4f}\n")
            f.write(
                f"Inference time (posterior) 10k: {run_time_posterior_inference:.4f}\n"
            )


if __name__ == "__main__":
    import config

    for seed in config.seeds:
        for simulation_budget in [2000, 6000, 10000]:
            two_moons_npe_c(
                simulation_budget=simulation_budget,
                seed=seed,
                prior=two_moons_wiqvist_torch.prior_torch,
                simulator=two_moons_wiqvist_torch.simulator_torch,
                num_posterior_samples=config.num_posterior_samples,
                dir_prefix="simulator_1/",
            )

            two_moons_npe_c(
                simulation_budget=simulation_budget,
                seed=seed,
                prior=two_moons_wiqvist_torch.prior_torch,
                simulator=sbibm.get_task("two_moons").get_simulator(),
                num_posterior_samples=config.num_posterior_samples,
                dir_prefix="simulator_2/",
            )
