import numpy as np

def default_mapfunc(theta, p):
    ang = -np.pi / 4.0
    c = np.cos(ang)
    s = np.sin(ang)  # this is the scaling
    z0 = c * theta[0] - s * theta[1]
    z1 = s * theta[0] + c * theta[1]
    return p + np.array([-np.abs(z0), z1])

def default_mapfunc_inverse_numpy(theta, x):
    ang = -np.pi / 4
    c = np.cos(ang)
    s = np.sin(ang)
    z0 = c * theta[0] - s * theta[1]
    z1 = s * theta[0] + c * theta[1]
    return x - np.array([-np.abs(z0), z1])

class TwoMoons():
    def __init__(self, mean_radius=1.0, sd_radius=0.1, baseoffset=1.0, fixed_data=True):
        self.mean_radius = mean_radius
        self.sd_radius = sd_radius
        self.baseoffset = baseoffset
        self.fixed_data = fixed_data  
        self.mapfunc = default_mapfunc
        self.mapfunc_inverse = default_mapfunc_inverse_numpy

    def likelihood(self, param, x, log=True):

        p = default_mapfunc_inverse(param, x)
        # assert p.size == 2, "not yet implemented for non-bijective map functions"
        u = p[0] - self.baseoffset
        v = p[1]

        if u < 0.0:  # invalid x for this theta
            return -np.array([float("Inf")]) if log else np.array([0.0])
        r = np.sqrt(u.pow(2) + v.pow(2))
        L = -0.5 * ((r - self.mean_radius) / self.sd_radius).pow(2) - 0.5 * np.log(2 * np.pi_tensor *
                                                                                   self.sd_radius ** 2)
        return L if log else np.exp(L)

    def gen_single(self, param):

        param = np.asarray(param).reshape(-1)
        a = np.pi * (np.random.uniform() - 0.5)
        r = self.mean_radius + np.random.normal() * self.sd_radius
        p = np.array([r * np.cos(a) + self.baseoffset, r * np.sin(a)])
        return self.mapfunc(param, p)

    def gen_posterior_samples(self, obs,
                              n_samples=1):  

        ang = -np.pi / 4.0
        c = np.cos(-ang)
        s = np.sin(-ang)

        theta = np.zeros((n_samples, 2))
        for i in range(n_samples):
            p = self.gen_single(np.zeros(2))  # ['data']
            q = np.zeros(2)
            q[0] = p[0] - obs[0]
            q[1] = obs[1] - p[1]

            if np.random.rand() < 0.5:
                q[0] = -q[0]

            theta[i, 0] = c * q[0] - s * q[1]
            theta[i, 1] = s * q[0] + c * q[1]
        return theta

    def model_sim(self, theta, dim=2):
        n = theta.shape[0]
        x_samples = np.zeros((n, dim))
        for i in range(n):
            x_samples[i,] = self.gen_single(theta[i, :])
        return x_samples

    def model_sim_numpy(self, theta, dim=2):
        return self.model_sim(theta, dim)