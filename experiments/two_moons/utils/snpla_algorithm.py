import torch
import copy
import math


def inference_snpla(
    flow_lik,
    flow_post,
    prior,
    simulator,
    optimizer_lik,
    optimizer_post,
    decay_rate_post,
    x_o,
    x_o_batch_post,
    dim_post,
    prob_prior,
    nbr_lik,
    nbr_epochs_lik,
    nbr_post,
    nbr_epochs_post,
    batch_size,
    batch_size_post,
    decay_rate_lik=0,
    epochs_hot_start=10,
    validation_fraction=0.1,
    early_stopping=True,
    stop_after_epochs=20,
):
    """
    Runs the snpla method

    :param flow_lik: (untrained) flow likelihood model
    :param flow_post: (untrained) flow posterior model
    :param prior: a pytorch distribution
    :param simulator: functions that simulations data for given theta returns a tensor of shape nxsim_data
    :param optimizer_lik: optimizer for the flow likelihood model
    :param optimizer_post: optimizer for the flow posterior model
    :param decay_rate_post: decay rate for the exponential decay of the lr for the posterior model
    :param x_o: observed data set should be of shape 1xd (where d is the dim of the obs. data)
    :param x_o_batch_post: data set of size batch_sizexd (where d is the dim of the obs. data)
    :param dim_post: dimension of the posterior dist
    :param prob_prior: list of length iterations with the prob. for prior in the mixture dist
    :param nbr_lik: list of length `iterations` with the number of model sims to use for each iteration
    :param nbr_epochs_lik: list of length `iterations` with the number of epochs for training the likelihood model for each iter
    :param nbr_post: list of length `iterations` with the number of samples from the current posterior model to use for each iteration
    :param nbr_epochs_post: list of length `iterations` with the number of epochs for training the posterior model for each iter
    :param batch_size: mini-batch size for training the likelihood model
    :param batch_size_post: mini-batch size for training the posterior model
    :param epochs_hot_start: number of epochs for the hot-start procedure (default value 10)
    :param validation_fraction: fraction of data used for validation (default value 0.1)
    :param early_stopping: use early-stopping when val data is not improving (default value True)
    :param stop_after_epochs: nbr of epochs to wait for improvment in val data (default value 20)
    :return models_lik - list of length `iterations` with the likelihood flow model obtained after each iteration,
    :return models_post - list of length `iterations` with the posterior flow model obtained after each iteration
    """

    nbr_iter = len(prob_prior)

    print("start full training")

    models_lik = []
    models_post = []

    scheduler_post = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer_post, gamma=decay_rate_post
    )
    scheduler_lik = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer_lik, gamma=decay_rate_lik
    )

    x_full_list = []
    theta_full_list = []

    for i in range(nbr_iter):
        # decay post lr
        if i >= 1 and decay_rate_post > 0:
            scheduler_post.step()

        if i >= 1 and decay_rate_lik > 0:
            scheduler_lik.step()

        # print iter info
        print("Iteration: " + str(i + 1))
        print("optimizer_post_lr: " + str(scheduler_post.get_last_lr()))
        print("optimizer_lik_lr: " + str(scheduler_lik.get_last_lr()))
        print("prob_prior: " + str(prob_prior[i]))

        nbr_lik_prior = int(prob_prior[i] * nbr_lik[i])
        nbr_like_post = int((1 - prob_prior[i]) * nbr_lik[i])
        print("nbr_like_prior: " + str(nbr_lik_prior))
        print("nbr_like_post: " + str(nbr_like_post))
        print("nbr_like_tot: " + str(nbr_lik_prior + nbr_like_post))

        theta_prior = prior.sample(sample_shape=(nbr_lik_prior,))

        if (
            nbr_like_post == 0
        ):  # this is to avoid concatunate a tensor with grad to the theta tensor
            theta_full = theta_prior
        else:
            theta_post = flow_post.sample(nbr_like_post, context=x_o)  # .reshape(1,dim)
            theta_post = theta_post.reshape((nbr_like_post, dim_post))
            theta_full = torch.cat([theta_prior, theta_post.detach()], dim=0)

            # this should not be needed if the flow models are set up correctly
            # # check if we have nans, e.g. remove thetas that are outside of prior
            # theta_prior_check = prior.log_prob(theta_post)
            # idx_save = (~torch.isinf(theta_prior_check)).nonzero(as_tuple=False)  # .nonzero(as_tuple=False)

            # print(idx_save.shape[0])

            # if idx_save.shape[0] > 0:
            #    theta_post = theta_post[idx_save.reshape(-1), :]  # remove ev. nans
            #    theta_full = torch.cat([theta_prior, theta_post.detach()], dim=0)  # add theta_prior and post
            # else:
            #    theta_full = theta_prior

        # run model sim
        x_full = simulator(theta_full)

        # append new likelihood training data to list
        x_full_list.append(x_full)
        theta_full_list.append(theta_full)

        # convert list to tensor
        x_full_like_train = torch.cat(x_full_list)
        theta_full_like_train = torch.cat(theta_full_list)

        print("Info on training data for like model:")
        print(x_full_like_train.shape)
        print(theta_full_like_train.shape)

        # update likelihood model
        _train_like(
            x_full_like_train,
            theta_full_like_train,
            nbr_epochs_lik[i],
            batch_size,
            flow_lik,
            optimizer_lik,
            validation_fraction,
            early_stopping,
            stop_after_epochs,
        )

        # update posterior model

        # 2' step: train posterior model from prior predictive first, only used to get a hot start
        if i == 0:
            _train_post_prior_pred(
                x_full_like_train,
                theta_full_like_train,
                epochs_hot_start,
                batch_size,
                flow_post,
                optimizer_post,
                validation_fraction,
            )
            # models_post.append(copy.deepcopy(flow_post))

        # Sample training data from posterior

        _train_post_sim_fly(
            nbr_post[i],
            nbr_epochs_post[i],
            batch_size_post,
            flow_post,
            flow_lik,
            optimizer_post,
            prior,
            x_o_batch_post,
            dim_post,
            x_o,
            validation_fraction,
            early_stopping,
            stop_after_epochs,
        )

        # save trained model for each iter
        models_lik.append(copy.deepcopy(flow_lik))
        models_post.append(copy.deepcopy(flow_post))

    return models_lik, models_post


def _train_like(
    x,
    theta,
    epochs,
    batch_size,
    flow_lik,
    optimizer_lik,
    validation_fraction,
    early_stopping,
    stop_after_epochs,
):
    print("start update likelihood model")

    x, theta, x_eval, theta_eval = _split_train_eval(x, theta, validation_fraction)

    losses_eval = []

    nbr_waited = 0
    current_lowest_eval_loss = float("inf")
    loss_start = float("inf")

    for e in range(epochs):  # this should be a while loop
        # print("--")
        # print(nbr_waited)
        # print(current_lowest_eval_loss)

        # run eval
        with torch.no_grad():
            loss_eval = -(flow_lik.log_prob(inputs=x_eval, context=theta_eval)).mean()
            losses_eval.append(loss_eval.item())

        # run early-stopping
        if early_stopping and e >= 1:  # TODO maybe this should be >= 10??
            nbr_waited, current_lowest_eval_loss = _early_stopping_check(
                losses_eval[e],
                losses_eval[e - 1],
                current_lowest_eval_loss,
                nbr_waited,
                stop_after_epochs,
                e,
            )
            if nbr_waited is None:
                return None

        loss_e = 0

        permutation = torch.randperm(x.size()[0])

        for i in range(0, x.size()[0], batch_size):
            optimizer_lik.zero_grad()

            indices = permutation[i : i + batch_size]
            input_x_batch = x[indices, :]
            input_theta_batch = theta[indices, :]

            loss = -(
                flow_lik.log_prob(inputs=input_x_batch, context=input_theta_batch)
            ).mean()

            loss_e = loss_e + loss.item()

            loss.backward()
            optimizer_lik.step()

        # if e == 0:
        #    loss_start = loss_e / (x.size()[0] / batch_size)
        # if e > 0 and loss_e / (x.size()[0] / batch_size) < loss_start/2:
        #    print("Early-stopping: loss halfed, do not imporve likelihood model more this iter")
        #    return None

        _print_update(e, loss_e / (x.size()[0] / batch_size), losses_eval[-1])

    return None


def _train_post_prior_pred(
    x, theta, epochs, batch_size, flow_post, optimizer_post, validation_fraction
):
    print("start update posterior model from prior pred - hot start")

    # TODO select random eval for x and theta

    x, theta, x_eval, theta_eval = _split_train_eval(x, theta, validation_fraction)

    losses_eval = []

    for e in range(epochs):  # this should be a while loop
        # run eval
        with torch.no_grad():
            loss_eval = -(flow_post.log_prob(inputs=theta_eval, context=x_eval)).mean()
            losses_eval.append(loss_eval.item())

        loss_e = 0

        permutation = torch.randperm(x.size()[0])

        for i in range(0, x.size()[0], batch_size):
            optimizer_post.zero_grad()

            indices = permutation[i : i + batch_size]
            input_x_batch = x[indices, :]
            input_theta_batch = theta[indices, :]

            loss = -(
                flow_post.log_prob(inputs=input_theta_batch, context=input_x_batch)
            ).mean()

            loss_e = loss_e + loss.item()

            loss.backward()
            optimizer_post.step()

        _print_update(e, loss_e / (x.size()[0] / batch_size), losses_eval[-1])

    return None


def _train_post_sim_fly(
    nbr_post,
    epochs,
    batch_size,
    flow_post,
    flow_lik,
    optimizer_post,
    prior,
    x_o_batch_post,
    dim_post,
    x_o,
    validation_fraction,
    early_stopping,
    stop_after_epochs,
):
    print("start update posterior model")

    # with torch.no_grad():
    #    for param in flow_post.parameters():
    #        param.add_(torch.randn(param.size()) * 0.01)

    losses_eval = []

    nbr_waited = 0
    current_lowest_eval_loss = float("inf")

    # create x_o data set for val
    nbr_val = int(nbr_post * validation_fraction)
    x_o_val = torch.zeros((nbr_val, x_o_batch_post.shape[1]))
    for i in range(nbr_val):
        x_o_val[i, :] = x_o_batch_post[0, :]

    # calc noise_eval
    # nbr_obs_eval = x_o_val.shape[0]
    # embedded_context = flow_post._embedding_net(x_o)  # .reshape(1,dim)
    # noise_eval, log_prob_eval = flow_post._distribution.sample_and_log_prob(
    #    nbr_obs_eval, context=embedded_context
    # )

    # noise_eval = noise_eval.reshape((nbr_obs_eval, dim_post))
    # logbase_post_eval = log_prob_eval.reshape((nbr_obs_eval))

    # create noise for batches
    # noise_batches = []
    # log_prob_batches = []

    # for i in range(nbr_post // batch_size):
    #    embedded_context = flow_post._embedding_net(x_o)  # .reshape(1,dim)
    #    noise, log_prob = flow_post._distribution.sample_and_log_prob(
    #        batch_size, context=embedded_context
    #    )

    #    noise_batches.append(noise.reshape((batch_size, dim_post)))
    #    log_prob_batches.append(log_prob.reshape((batch_size)))

    for e in range(epochs):
        # set new order of the batches
        idx_batches = torch.randperm(nbr_post // batch_size)

        loss_e = 0

        # TODO: Maybe we want to use the same random numbers in all epochs, st the varaiability of the post model is reduced

        # run eval from new sims
        with torch.no_grad():
            nbr_obs_eval = x_o_val.shape[0]
            embedded_context = flow_post._embedding_net(x_o)  # .reshape(1,dim)
            noise_eval, log_prob_eval = flow_post._distribution.sample_and_log_prob(
                nbr_obs_eval, context=embedded_context
            )

            noise_eval = noise_eval.reshape((nbr_obs_eval, dim_post))
            logbase_post_eval = log_prob_eval.reshape((nbr_obs_eval))
            loss_eval = _calc_loss_post_training(
                flow_post, flow_lik, prior, noise_eval, logbase_post_eval, x_o_val
            )

            losses_eval.append(loss_eval.item())

        # run early-stopping
        if early_stopping and e >= 1:
            nbr_waited, current_lowest_eval_loss = _early_stopping_check(
                losses_eval[e],
                losses_eval[e - 1],
                current_lowest_eval_loss,
                nbr_waited,
                stop_after_epochs,
                e,
            )
            if nbr_waited is None:
                return None

        for i in range(nbr_post // batch_size):
            embedded_context = flow_post._embedding_net(x_o)  # .reshape(1,dim)
            noise, log_prob = flow_post._distribution.sample_and_log_prob(
                batch_size, context=embedded_context
            )

            noise_batch = noise.reshape((batch_size, dim_post))
            logbase_post_batch = log_prob.reshape((batch_size))

            optimizer_post.zero_grad()

            # noise_batch = noise_batches[idx_batches[i]]
            # logbase_post_batch = log_prob_batches[idx_batches[i]]

            loss = _calc_loss_post_training(
                flow_post,
                flow_lik,
                prior,
                noise_batch,
                logbase_post_batch,
                x_o_batch_post,
            )

            loss_e = loss_e + loss.item()

            loss.backward()
            optimizer_post.step()

        _print_update(e, loss_e / (nbr_post // batch_size), losses_eval[-1])

    return None


def _calc_loss_post_training(
    flow_post, flow_lik, prior, noise_batch, logbase_post_batch, x_o_batch_post
):
    embedded_context = flow_post._embedding_net(x_o_batch_post)  # .reshape(1,dim)

    theta_batch, logabsdet_post = flow_post._transform.inverse(
        noise_batch, context=embedded_context
    )

    # TODO this step can be made much easier to follow by directly using sample_and_log_prob
    #  (which is ok since we do not need the the pdf of the base dist)

    # loss_comp_post = logbase_post_batch - logabsdet_post
    loss_comp_post = -logabsdet_post

    loss_comp_lik = flow_lik._log_prob(x_o_batch_post, context=theta_batch)

    if isinstance(prior, torch.distributions.uniform.Uniform):
        # uniform prior, we do not have to include the log pdf of the prior since the posterior will have the correct
        # support
        return (loss_comp_post - loss_comp_lik).mean()
    else:
        # some other prior dist, include prior log pdf
        loss_comp_prior = prior.log_prob(theta_batch)
        return (loss_comp_post - loss_comp_lik - loss_comp_prior).mean()

    # loss_tot = (loss_comp_post - loss_comp_lik - loss_comp_prior).mean()

    # print("---")
    # print(prior)
    # print(prior.base_dist)
    # print(isinstance(prior.base_dist, torch.distributions.uniform.Uniform))

    # if torch.isinf(loss_tot):
    #    print("--")
    #    idx = (torch.isinf(loss_comp_prior)).nonzero(as_tuple=False)
    #    print(theta_batch[idx, :])
    #    print(loss_comp_post.mean())
    #    print(loss_comp_lik.mean())
    #    print(loss_comp_prior.mean())
    #    print(loss_comp_prior[idx])

    # return loss_tot  # (loss_comp_post - loss_comp_lik - loss_comp_prior).mean()


# TODO have to check how this playes along with negative losses
def _early_stopping_check(
    loss_new, loss_old, current_lowest_eval_loss, nbr_waited, stop_after_epochs, epoch
):
    # update current_lowest_eval_loss and nbr_waited
    if loss_new > loss_old:  # early-stopping eval loss is not improving
        if (
            loss_old < current_lowest_eval_loss
        ):  # updated current_lowest_eval_loss if loss_old is better
            current_lowest_eval_loss = loss_old
        nbr_waited = nbr_waited + 1  # increase nbr_waited
    elif (
        loss_new > current_lowest_eval_loss
    ):  # loss_new is not better than current_lowest_eval_loss, increase
        # nbr_waited
        nbr_waited = nbr_waited + 1
    elif (
        loss_new < current_lowest_eval_loss
    ):  # loss_new is better than current_lowest_eval_loss, reset nbr_waited
        current_lowest_eval_loss = loss_new
        nbr_waited = 0  # reset nbr_waited

    # end training if we should not wait longer
    if nbr_waited >= stop_after_epochs:  # return none
        _print_early_stopping_info(epoch)
        return None, current_lowest_eval_loss

    return nbr_waited, current_lowest_eval_loss


def _print_early_stopping_info(epochs):
    print("Early-stopping. Training converged after " + str(epochs) + " epochs.")


def _print_update(epoch, train_loss, eval_loss):
    print(
        "Epoch: "
        + str(epoch)
        + ", loss (training): "
        + str(round(train_loss, 4))
        + ", loss (eval): "
        + str(round(eval_loss, 4))
    )


def _split_train_eval(x, theta, validation_fraction):
    permutation_eval = torch.randperm(x.size()[0])
    nbr_eval = int(x.size()[0] * validation_fraction)
    idx_eval = permutation_eval[:nbr_eval]
    idx_train = permutation_eval[nbr_eval:]

    x_eval = x[idx_eval, :]
    theta_eval = theta[idx_eval, :]

    x = x[idx_train, :]
    theta = theta[idx_train, :]

    return x, theta, x_eval, theta_eval


def calc_prob_prior(iterations, lam):
    """
    Calculates the alpha prob for the mixture as

    $$\alpha_i = \exp(-ki), \quad i = 0,1,\ldots,iterations$$


    :param iterations: nbr of iterations for snpla
    :param lam: decay rate
    :return: list of length iterations with the prob. for prior in the mixture dist.
    """
    return list(map(lambda x: math.exp(-lam * x), range(iterations)))
