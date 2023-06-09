import sbibm
import torch
import numpy as np
import time

import two_moons_wiqvist_torch
from utils.snpla_algorithm import inference_snpla, calc_prob_prior
from utils.sbi_network_setup import set_up_networks
from config import torch_device

torch.set_default_device(torch_device)

x_o = torch.tensor([0, 0], device=torch_device, dtype=torch.float32)
x_o = x_o.flatten()


def two_moons_snpla(num_rounds, seed, prior, simulator, dir_prefix=""):
    flow_lik, flow_post = set_up_networks(seed)
    optimizer_lik = torch.optim.Adam(flow_lik.parameters(), lr=0.001, weight_decay=0.0)
    optimizer_post = torch.optim.Adam(
        flow_post.parameters(), lr=0.001, weight_decay=0.0
    )
    decay_rate_post = 0.9
    prob_prior_decay_rate = 0.7
    prob_prior = calc_prob_prior(num_rounds, prob_prior_decay_rate)

    nbr_lik = [1000 for _ in range(num_rounds)]
    nbr_epochs_lik = [25 for _ in range(num_rounds)]
    batch_size = 2000
    batch_size_post = 20000
    nbr_post = [60000 for _ in range(num_rounds)]
    nbr_epochs_post = [75 for _ in range(num_rounds)]

    x_o_batch_post = torch.zeros(batch_size_post, 2, device=torch_device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    start = time.time()

    models_lik, models_post = inference_snpla(
        flow_lik,
        flow_post,
        prior,
        simulator,
        optimizer_lik,
        optimizer_post,
        decay_rate_post,
        x_o.reshape(1, 2),
        x_o_batch_post,
        2,
        prob_prior,
        nbr_lik,
        nbr_epochs_lik,
        nbr_post,
        nbr_epochs_post,
        batch_size,
        batch_size_post,
    )
    end = time.time()
    run_time = end - start

    start = time.time()

    for i in range(num_rounds):
        theta_trained = models_post[i].sample(1000, context=x_o.reshape(1, 2))
        theta_trained = theta_trained.reshape(1000, 2)

        np.savetxt(
            f"output/{dir_prefix}snpla_post_round{i + 1}_seed{seed}.csv",
            theta_trained.detach().cpu().numpy(),
            delimiter=",",
        )

    end = time.time()
    run_time_posterior_inference = (end - start) / num_rounds

    # Posterior Predictive inference
    start = time.time()
    for i in range(num_rounds):
        theta_trained = models_post[i].sample(1000, context=x_o.reshape(1, 2))
        theta_trained = theta_trained.reshape((1000, 2))

        x_trained = models_lik[i].sample(1, context=theta_trained)
        x_trained = x_trained.reshape((1000, 2))

        np.savetxt(
            f"output/{dir_prefix}snpla_post_pred_round{i + 1}_seed{seed}.csv",
            x_trained.detach().cpu().numpy(),
            delimiter=",",
        )

    end = time.time()
    run_time_posterior_predictive_inference = (end - start) / num_rounds

    with open(f"output/{dir_prefix}snpla_metadata_seed{seed}.txt", "w") as f:
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
        two_moons_snpla(
            num_rounds=config.num_rounds,
            seed=seed,
            prior=two_moons_wiqvist_torch.prior_torch,
            simulator=two_moons_wiqvist_torch.simulator_torch_batched,
            dir_prefix="simulator_1/",
        )

        two_moons_snpla(
            num_rounds=config.num_rounds,
            seed=seed,
            prior=two_moons_wiqvist_torch.prior_torch,
            simulator=sbibm.get_task("two_moons").get_simulator(),
            dir_prefix="simulator_2/",
        )
