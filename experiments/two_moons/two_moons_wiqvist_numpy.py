import numpy as np

mean_radius = 1.0
sd_radius = 0.1
baseoffset = 1.0


def prior_numpy(lb, ub, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(low=-2.0, high=2.0, size=2)


def simulator_numpy(theta):
    theta = theta.reshape(-1)
    a = np.array(np.pi * (np.random.random(1) - 0.5))
    r = mean_radius + np.random.normal(loc=0, scale=1, size=1) * sd_radius
    p = np.array([r * np.cos(a) + baseoffset, r * np.sin(a)])
    ang = np.array([-np.pi / 4.0])
    c = np.cos(ang)
    s = np.sin(ang)
    z0 = c * theta[0] - s * theta[1]
    z1 = s * theta[0] + c * theta[1]
    return p + np.array([-np.abs(z0), z1])


def simulator_numpy_batched(theta):
    n_sim, data_dim = theta.shape
    x_samples = np.zeros((n_sim, data_dim))

    for i in range(n_sim):
        x_samples[i, ] = simulator_numpy(theta[i, ]).flatten()

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
