import os
from typing import Tuple, Union, List, Dict, Any
import random

import numpy as np
import torch

def generate_dir_name(options:object,perturbation:Union[Tuple[float,float],None]=None):
    if perturbation is None:
        sigma, scale = options.vel_sigma, options.vel_scale
        return f'orgnl_sigma_{sigma}_scale_{scale}'
    else:
        sigma, scale = perturbation
        return f'prtrb_sigma_{sigma}_scale_{scale}'
    
def generate_run_ID(options):
    ''' 
    Create a unique run ID from the most relevant
    parameters.
    '''
    params = [
        'steps', str(options.sequence_length),
        'batch', str(options.batch_size),
        options.RNN_type,
        str(options.Ng),
        options.activation,
        'rf', str(options.place_cell_rf),
        'DoG', str(options.DoG),
        'periodic', str(options.periodic),
        'lr', str(options.learning_rate),
        'decay', str(options.weight_decay),
        'vsgm', str(options.vel_sigma),
        'vscl', str(options.vel_scale),
        'hsgm', str(options.hid_sigma),
        'hscl', str(options.hid_scale),
        'seed', str(options.seed),
        ]
    separator = '_'
    run_ID = separator.join(params)
    run_ID = run_ID.replace('.', '')

    return run_ID

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True