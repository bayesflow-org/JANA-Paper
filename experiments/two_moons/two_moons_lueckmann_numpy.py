import numpy as np


def simulator_numpy(theta, rng=None):
    # Use default RNG, if None specified
    if rng is None:
        rng = np.random.default_rng()

    # Generate noise
    alpha = rng.uniform(low=-0.5 * np.pi, high=0.5 * np.pi)
    r = rng.normal(loc=0.1, scale=0.01)

    # Forward process
    rhs1 = np.array([r * np.cos(alpha) + 0.25, r * np.sin(alpha)])
    rhs2 = np.array(
        [
            -np.abs(theta[0] + theta[1]) / np.sqrt(2.0),
            (-theta[0] + theta[1]) / np.sqrt(2.0),
        ]
    )

    return rhs1 + rhs2


def simulator_numpy_batched(theta):
    n_sim, data_dim = theta.shape
    x_samples = np.zeros((n_sim, data_dim))

    for i in range(n_sim):
        x_samples[i,] = simulator_numpy(theta[i,])

    return x_samples


def analytic_posterior_numpy(x_o, n_samples=1):
    ang = -np.pi / 4.0
    c = np.cos(-ang)
    s = np.sin(-ang)

    theta = np.zeros((n_samples, 2))

    for i in range(n_samples):
        p = simulator_numpy(np.zeros(2))
        q = np.zeros(2)

        q[0] = p[0] - x_o[0]
        q[1] = x_o[1] - p[1]

        if np.random.rand() < 0.5:
            q[0] = -q[0]

        theta[i, 0] = c * q[0] - s * q[1]
        theta[i, 1] = s * q[0] + c * q[1]

    return theta
