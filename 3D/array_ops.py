import numpy as np
import torch as tc
import os
import dxchange

def initialize_guess_3d(dev, 
                        ini_kind, 
                        n_element, 
                        sample_size_n, 
                        sample_height_n, 
                        init_const = 0.5, 
                        ini_rand_amp = 0.1):
    
    if ini_kind == "rand": # Offset + random numbers between 0 and 1 based on uniform distribution (each number is equally likely to be chosen)
        X = init_const + ini_rand_amp * tc.rand(n_element, sample_height_n, sample_size_n, sample_size_n, device=dev)
        X = tc.clamp(X, 0, float('inf'))

    elif ini_kind == "randn": # Offset + random numbers based on standardnormal distribution (mean = 0, standard deviation = 1)
        X = init_const + ini_rand_amp * tc.randn(n_element, sample_height_n, sample_size_n, sample_size_n, device=dev)
        X = tc.clamp(X, 0, float('inf'))

    elif ini_kind == "const":
        X = init_const + tc.zeros(n_element, sample_height_n, sample_size_n, sample_size_n, device=dev)

    else:
        print("Please specify the correct type of the initialization condition.")

    return X