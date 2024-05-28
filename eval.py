import numpy as np
import torch
import os
import csv
import time

from tqdm import tqdm
from matplotlib import pyplot as plt
import scipy

from utils import generate_run_ID
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer
from utils import generate_dir_name, compute_ratemaps, plot_ratemaps, seed_everything

import argparse

def plot_trajectory(place_cells,options,model,trajectory_generator,perturbation=None):
    inputs, pos, _ = trajectory_generator.get_test_batch(batch_size=100)
    pos = pos.cpu()
    pred_pos = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()
    us = place_cells.us.cpu()

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    for i in range(5):
        plt.plot(pos[:,i,0], pos[:,i,1], c='black', label='True position', linewidth=2)
        plt.plot(pred_pos[:,i,0], pred_pos[:,i,1], '.-',
                c='C1', label='Decoded position')
        if i==0:
            plt.legend()
    plt.scatter(us[:,0], us[:,1], s=20, alpha=0.5, c='lightgrey')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-options.box_width/2,options.box_width/2])
    plt.ylim([-options.box_height/2,options.box_height/2])
    plt.savefig(f'images/{options.run_ID}/{generate_dir_name(options,perturbation)}/sim_traj_decode.pdf')
    plt.clf()
    plt.close()

def plot_place_cells(place_cells,options,model,trajectory_generator,perturbation=None):
    inputs, _, pc_outputs = trajectory_generator.get_test_batch(batch_size=100)
    preds = model.predict(inputs)
    preds = preds.reshape(-1, options.Np).detach().cpu()
    pc_outputs = model.softmax(pc_outputs).reshape(-1, options.Np).cpu()
    pc_pred = place_cells.grid_pc(preds[:100])
    pc = place_cells.grid_pc(pc_outputs[:100])

    plt.figure(figsize=(16,4))
    for i in range(8):
        plt.subplot(2,8,i+9)
        plt.imshow(pc_pred[2*i], cmap='jet')
        if i==0:
            plt.ylabel('Predicted')
        plt.axis('off')
        
    for i in range(8):
        plt.subplot(2,8,i+1)
        plt.imshow(pc[2*i], cmap='jet', interpolation='gaussian')
        if i==0:
            plt.ylabel('True')
        plt.axis('off')
        
    plt.suptitle('Place cell outputs', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"images/{options.run_ID}/{generate_dir_name(options,perturbation)}/place_cell.pdf")
    plt.clf()
    plt.close()

def plot_grid_cells(options,model,trajectory_generator,res,n_avg,perturbation=None):
    Ng = options.Ng
    options.batch_size = 200
    activations, _, _, _, counts = compute_ratemaps(model,trajectory_generator,options,res=res,n_avg=n_avg,Ng=Ng)

    np.save(f'data/{options.run_ID}/{generate_dir_name(options,perturbation)}/activs.npy',activations)
    np.save(f'data/{options.run_ID}/{generate_dir_name(options,perturbation)}/counts.npy',counts)

    n_plot = 256
    plt.figure(figsize=(16,4*n_plot//8**2))
    rm_fig = plot_ratemaps(activations, n_plot, smooth=True)
    plt.imshow(rm_fig)
    plt.axis('off')
    plt.savefig(f"images/{options.run_ID}/{generate_dir_name(options,perturbation)}/grid_cell.pdf")
    plt.clf()
    plt.close()

    return activations

def worker(act):
    from scores import GridScorer
    scorer = GridScorer(0)
    autocorr, fpautocorr = scorer.calc_score_rot(act)
    _, _, speccorr, fpspeccorr = scorer.calc_score_fourier(act, new_res=255)
    speccorr_new, fpspeccorr_new = scorer.calc_score_fourier_new(act, new_res=255)
    return speccorr, fpspeccorr, speccorr_new, fpspeccorr_new, autocorr, fpautocorr

if __name__=='__main__':
    import multiprocessing as mp
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',
                        default='models/',
                        type=str,
                        help='directory to save trained models')
    parser.add_argument('--n_epochs',
                        default=100,
                        type=int,
                        help='number of training epochs')
    parser.add_argument('--n_steps',
                        default=1000,
                        type=int,
                        help='batches per epoch')
    parser.add_argument('--batch_size',
                        default=200,
                        type=int,
                        help='number of trajectories per batch')
    parser.add_argument('--sequence_length',
                        default=20,
                        type=int,
                        help='number of steps in trajectory')
    parser.add_argument('--learning_rate',
                        default=1e-4,
                        type=float,
                        help='gradient descent learning rate')
    parser.add_argument('--Np',
                        default=512,
                        type=int,
                        help='number of place cells')
    parser.add_argument('--Ng',
                        default=4096,
                        type=int,
                        help='number of grid cells')
    parser.add_argument('--place_cell_rf',
                        default=0.12,
                        type=float,
                        help='width of place cell center tuning curve (m)')
    parser.add_argument('--surround_scale',
                        default=2,
                        type=int,
                        help='if DoG, ratio of sigma2^2 to sigma1^2')
    parser.add_argument('--RNN_type',
                        default='RNN',
                        help='RNN or LSTM')
    parser.add_argument('--activation',
                        default='relu',
                        help='recurrent nonlinearity')
    parser.add_argument('--weight_decay',
                        default=1e-4,
                        type=float,
                        help='strength of weight decay on recurrent weights')
    parser.add_argument('--vel_sigma',
                        default=0.0,
                        type=float,
                        help='standard deviation of noise in velocity input')
    parser.add_argument('--vel_scale',
                        default=1.0,
                        type=float,
                        help='attenuation of velocity input')
    parser.add_argument('--hid_sigma',
                        default=0.0,
                        type=float,
                        help='standard deviation of noise in neurons')
    parser.add_argument('--hid_scale',
                        default=1.0,
                        type=float,
                        help='attenuation of neurons')
    parser.add_argument('--DoG',
                        default=True,
                        help='use difference of gaussians tuning curves')
    parser.add_argument('--periodic',
                        default=False,
                        help='trajectories with periodic boundary conditions')
    parser.add_argument('--box_width',
                        default=2.2,
                        type=float,
                        help='width of training environment')
    parser.add_argument('--box_height',
                        default=2.2,
                        type=float,
                        help='height of training environment')
    parser.add_argument('--core_id',
                        default=0,
                        type=int,
                        help='GPU device ID')
    parser.add_argument('--seed',
                        default=0,
                        type=int,
                        help='seed')
    parser.add_argument('--res',
                        default=35,
                        type=int,
                        help='resolution for the rate map')
    parser.add_argument('--prtrb_type',
                        default='activation',
                        type=str,
                        help='type of perturbation')

    options = parser.parse_args()
    seed_everything(options.seed)

    options.run_ID = generate_run_ID(options)
    options.device = torch.device(f'cuda:{options.core_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {options.device}')


    place_cells = PlaceCells(options)
    model = RNN(options, place_cells)

    ckpt_dir = os.path.join(options.save_dir, options.run_ID)
    ckpt_path = os.path.join(ckpt_dir, 'most_recent_model.pth')
    model.load_state_dict(torch.load(ckpt_path))
    model = model.to(options.device)
    model.eval()

    trajectory_generator = TrajectoryGenerator(options, place_cells)

    if options.prtrb_type == 'activation':
        prtrb_list = [(0.0, 1.0, 0.0, 1.0)]
        prtrb_list += [(vel_sigma, 1.0, 0.0, 1.0) for vel_sigma in np.linspace(0.02,0.2,10)]
        prtrb_list += [(0.0, 1.0, hid_sigma, 1.0) for hid_sigma in np.linspace(0.02,0.2,10)]
    elif options.prtrb_type == 'weight_noise':
        prtrb_list = [(0.0, 0.0)]
        prtrb_list += [(vswsgm, 0.0) for vswsgm in np.linspace(0.2,1.0,5)]
        prtrb_list += [(0.0, hswsgm) for hswsgm in np.linspace(0.01,0.05,5)]
    elif options.prtrb_type == 'weight_mask':
        prtrb_list = [(0.0, 0.0)]
        prtrb_list += [(vswthr, 0.0) for vswthr in np.linspace(0.1,1.0,10)]
        prtrb_list += [(0.0, hswthr) for hswthr in np.linspace(0.001,0.01,10)]

    for prtrb in prtrb_list:
        os.makedirs(f'images/{options.run_ID}/{generate_dir_name(options,prtrb)}',exist_ok=True)
        os.makedirs(f'data/{options.run_ID}/{generate_dir_name(options,prtrb)}/',exist_ok=True)

        start = time.time()

        with torch.no_grad():
            if options.prtrb_type=='activation' and prtrb is not None:
                model.vel_sigma = prtrb[0]
                model.vel_scale = prtrb[1]
                model.hid_sigma = prtrb[2]
                model.hid_scale = prtrb[3]
            elif options.prtrb_type=='weight_noise' and prtrb is not None:
                model.vel_stream.weight += prtrb[0] * torch.randn_like(model.vel_stream.weight)
                model.hid_stream.weight += prtrb[1] * torch.randn_like(model.hid_stream.weight)
            elif options.prtrb_type=='weight_mask' and prtrb is not None:
                model.vel_stream.weight[abs(model.vel_stream.weight)<prtrb[0]] = 0.0
                model.hid_stream.weight[abs(model.hid_stream.weight)<prtrb[1]] = 0.0

            plot_trajectory(place_cells,options,model,trajectory_generator,prtrb)

            plot_place_cells(place_cells,options,model,trajectory_generator,prtrb)

            activations = plot_grid_cells(options,model,trajectory_generator,res=options.res,n_avg=100,perturbation=prtrb)

        pool_args = [(act,) for act in activations]
        with mp.Pool(processes=32) as p:
            results = p.starmap(worker, pool_args)
        speccorr, fpspeccorr, speccorr_new, fpspeccorr_new, autocorr, fpautocorr = zip(*results)
        np.save(f'data/{options.run_ID}/{generate_dir_name(options,prtrb)}/speccorr.npy',speccorr)
        np.save(f'data/{options.run_ID}/{generate_dir_name(options,prtrb)}/fpspeccorr.npy',fpspeccorr)
        np.save(f'data/{options.run_ID}/{generate_dir_name(options,prtrb)}/speccorr_new.npy',speccorr_new)
        np.save(f'data/{options.run_ID}/{generate_dir_name(options,prtrb)}/fpspeccorr_new.npy',fpspeccorr_new)
        np.save(f'data/{options.run_ID}/{generate_dir_name(options,prtrb)}/autocorr.npy',autocorr)
        np.save(f'data/{options.run_ID}/{generate_dir_name(options,prtrb)}/fpautocorr.npy',fpautocorr)
        np.save(f'data/{options.run_ID}/{generate_dir_name(options,prtrb)}/activations.npy',activations)

        print(f'Total time: {time.time()-start}')