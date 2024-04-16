from typing import Tuple, Union, List, Dict, Any

def generate_dir_name(options:object,perturbation:Union[Tuple[float,float],None]=None):
    if perturbation is None:
        sigma, scale = options.vel_sigma, options.vel_scale
        return f'orgnl_sigma_{sigma}_scale_{scale}'
    else:
        sigma, scale = perturbation
        return f'prtrb_sigma_{sigma}_scale_{scale}'