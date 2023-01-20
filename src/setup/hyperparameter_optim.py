import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
import pickle

def generate_hyperparameter_configurations():
    #rng = np.random.RandomState(0)
    param_grid = {'lr':np.arange(-4,-5.5,-0.1).tolist(), 'wtdecay':np.arange(-3.5,-5.5,-0.1).tolist(), 'sm_reg_param':np.arange(-3.5,-5.5,-0.1).tolist()}
    #param_grid = {'lr':np.arange(-4,-5.5,-0.1).tolist(), 'sm_reg_param':np.arange(-3.5,-5.5,-0.1).tolist()}
    #param_list = list(ParameterGrid(param_grid))
    param_list = list(ParameterSampler(param_grid, n_iter=20, random_state=42))
    rounded_list = [dict((k, round(v,1)) for (k, v) in d.items()) for d in param_list]
    print(rounded_list)
    return rounded_list