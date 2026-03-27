import torch
import random
import numpy as np
import os

def generate_sampling_pattern(sampling_percentage,N, if_seed = False, seed = 42, is_line_sampling = False):

    if if_seed:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # backends
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    
    if is_line_sampling:
        num_lines = int(N * sampling_percentage)
        indices = torch.randperm(N)[:num_lines]
        pattern = np.zeros((N, N))
        pattern[indices, :] = 1
        return pattern
    
    else:
        indices = torch.randperm(N * N)[:int(N * N * sampling_percentage)]
        pattern = torch.zeros(N, N)
        pattern.view(-1)[indices] = 1
        return pattern.numpy()


def generate_local_averaging_filter(kernel_size=(5,5,), sigma = 1):
    
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size,  dtype=torch.float32,) for size in kernel_size]) #torch.meshgrid([torch.linspace(0, size - 1, 1000, dtype=torch.float32) for size in kernel_size],indexing='ij')
#
    for size, mgrid in zip(kernel_size, meshgrids):
        mean = (size - 1) / 2
        kernel *= torch.exp(-((mgrid - mean) / sigma) ** 2 / 2)

    # Make sure norm of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)
    return np.array(kernel)

    # kernel = np.ones(kernel_size)
    # kernel /= np.sum(kernel)

    # return kernel