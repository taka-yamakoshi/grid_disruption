import os
from typing import Tuple, Union, List, Dict, Any
import random
import matplotlib.pyplot as plt
import cv2

import numpy as np
import torch

def generate_dir_name(options:object,perturbation:Union[Tuple[float,float,float,float],None]=None):
    if perturbation is None:
        vsgm, vscl, hsgm, hscl = options.vel_sigma, options.vel_scale, options.hid_sigma, options.hid_scale
        return f'orgnl_vsgm_{vsgm:.2g}_vscl_{vscl:.2g}_hsgm_{hsgm:.2g}_hscl_{hscl:.2g}'
    else:
        vsgm, vscl, hsgm, hscl = perturbation
        return f'prtrb_vsgm_{vsgm:.2g}_vscl_{vscl:.2g}_hsgm_{hsgm:.2g}_hscl_{hscl:.2g}'
    
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

class Options(object):
    pass

def get_default_options():
    options = Options()
    options.save_dir ='models/'
    options.n_epochs = 100
    options.n_steps = 1000
    options.batch_size = 200
    options.sequence_length = 20
    options.learning_rate = 1e-4
    options.Np = 512
    options.Ng = 4096
    options.place_cell_rf = 0.12
    options.surround_scale = 2
    options.RNN_type = 'RNN'
    options.activation ='relu'
    options.weight_decay = 1e-4
    options.vel_sigma = 0.0
    options.vel_scale = 1.0
    options.hid_sigma = 0.0
    options.hid_scale = 1.0
    options.DoG = True
    options.periodic = False
    options.box_width = 2.2
    options.box_height = 2.2
    options.core_id = 0
    options.seed = 0
    return options

# mostly copied from https://github.com/ganguli-lab/grid-pattern-formation/blob/master/visualize.py
def compute_ratemaps(model, trajectory_generator, options, res=20, n_avg=None, Ng=512, idxs=None):
    '''Compute spatial firing fields'''

    if not n_avg:
        n_avg = 1000 // options.sequence_length

    if not np.any(idxs):
        idxs = np.arange(Ng)
    idxs = idxs[:Ng]

    g = np.zeros([n_avg, options.batch_size * options.sequence_length, Ng])
    pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])

    activations = np.zeros([Ng, res, res]) 
    counts  = np.zeros([res, res])

    for index in range(n_avg):
        inputs, pos_batch, _ = trajectory_generator.get_test_batch(batch_size=options.batch_size)
        g_batch = model.g(inputs).detach().cpu().numpy()
        
        pos_batch = np.reshape(pos_batch.cpu(), [-1, 2])
        g_batch = g_batch[:,:,idxs].reshape(-1, Ng)
        
        g[index] = g_batch
        pos[index] = pos_batch

        x_batch = (pos_batch[:,0] + options.box_width/2) / (options.box_width) * res
        y_batch = (pos_batch[:,1] + options.box_height/2) / (options.box_height) * res

        for i in range(options.batch_size*options.sequence_length):
            x = x_batch[i]
            y = y_batch[i]
            if x >=0 and x < res and y >=0 and y < res:
                counts[int(x), int(y)] += 1
                activations[:, int(x), int(y)] += g_batch[i, :]

    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]
                
    g = g.reshape([-1, Ng])
    pos = pos.reshape([-1, 2])

    # # scipy binned_statistic_2d is slightly slower
    # activations = scipy.stats.binned_statistic_2d(pos[:,0], pos[:,1], g.T, bins=res)[0]
    rate_map = activations.reshape(Ng, -1)

    return activations, rate_map, g, pos, counts

def concat_images(images, image_width, spacer_size):
    """ Concat image horizontally with spacer """
    spacer = np.ones([image_width, spacer_size, 4], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1:
            # Add spacer
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret


def concat_images_in_rows(images, row_size, image_width, spacer_size=4):
    """ Concat images in rows """
    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_width*column_size + (column_size-1)*spacer_size, 4],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size*row:column_size*row+column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size-1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret


def convert_to_colormap(im, cmap):
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def rgb(im, cmap='jet', smooth=True):
    cmap = plt.cm.get_cmap(cmap)
    np.seterr(invalid='ignore')  # ignore divide by zero err
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    if smooth:
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def plot_ratemaps(activations, n_plots, cmap='jet', smooth=True, width=16):
    images = [rgb(im, cmap, smooth) for im in activations[:n_plots]]
    rm_fig = concat_images_in_rows(images, n_plots//width, activations.shape[-1])
    return rm_fig