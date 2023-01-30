from functools import partial
import tensorflow as tf
import numpy as np


def maximum_mean_discrepancy(source_samples, target_samples, kernel = "gaussian", 
minimum=0., unbiased=False, squared=True):
    """ This Maximum Mean Discrepancy (MMD) loss is calculated with a number of different Gaussian or Inverse-Multiquadratic kernels.
    """

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    
    if kernel == "gaussian":
        kernel = partial(_gaussian_kernel_matrix, sigmas=sigmas)
    elif kernel == "inverse_multiquadratic":
        kernel = partial(_inverse_multiquadratic_kernel_matrix, sigmas=sigmas)
    else:
        print("Invalid kernel specified. Falling back to default Gaussian.")
        kernel = partial(_gaussian_kernel_matrix, sigmas=sigmas)
    
    if unbiased:
        loss_value = _mmd_kernel_unbiased(source_samples, target_samples, kernel=kernel)
    else:
        loss_value = _mmd_kernel(source_samples, target_samples, kernel=kernel)
        
        
    loss_value = tf.maximum(minimum, loss_value) 
    
    if squared:
        return loss_value
    else:
        return tf.math.sqrt(loss_value)
    
def _gaussian_kernel_matrix(x, y, sigmas):
    """ Computes a Gaussian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width :math:`\sigma_i`.
    Parameters
    ----------
    x :  tf.Tensor of shape (M, num_features)
    y :  tf.Tensor of shape (N, num_features)
    sigmas : list(float)
        List which denotes the widths of each of the gaussians in the kernel.
    Returns
    -------
    kernel: tf.Tensor
        RBF kernel of shape [num_samples{x}, num_samples{y}]
    """

    norm = lambda v: tf.math.reduce_sum(tf.square(v), 1)
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    kernel = tf.reshape(tf.math.reduce_sum(tf.exp(-s), 0), tf.shape(dist))
    return 

def _mmd_kernel(x, y, kernel=None):
    """ Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of the distributions of x and y.
    Parameters
    ----------
    x      : tf.Tensor of shape (num_samples, num_features)
    y      : tf.Tensor of shape (num_samples, num_features)
    kernel : callable, default: _gaussian_kernel_matrix
        A function which computes the kernel in MMD.
    Returns
    -------
    loss : tf.Tensor
        squared maximum mean discrepancy loss, shape (,)
    """

    loss = tf.math.reduce_mean(kernel(x, x))  
    loss += tf.math.reduce_mean(kernel(y, y))  
    loss -= 2 * tf.math.reduce_mean(kernel(x, y))
    return loss

def _mmd_kernel_unbiased(x, y, kernel=None):
    """ Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of the distributions of x and y.
    Parameters
    ----------
    x      : tf.Tensor of shape (num_samples, num_features)
    y      : tf.Tensor of shape (num_samples, num_features)
    kernel : callable, default: _gaussian_kernel_matrix
        A function which computes the kernel in MMD.
    Returns
    -------
    loss : tf.Tensor
        squared maximum mean discrepancy loss, shape (,)
    """
    m, n = x.shape[0], y.shape[0]
    loss = (1.0/(m*(m-1))) * tf.math.reduce_sum(kernel(x, x))  
    loss += (1.0/(n*(n-1))) * tf.math.reduce_sum(kernel(y, y)) 
    loss -= (2.0/(m*n)) * tf.math.reduce_sum(kernel(x, y)) 
    return loss