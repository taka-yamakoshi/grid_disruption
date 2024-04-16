import torch

from utils import generate_run_ID, seed_everything
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer

import argparse

if __name__ == '__main__':
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

    options = parser.parse_args()
    seed_everything(options.seed)

    options.run_ID = generate_run_ID(options)
    options.device = torch.device(f'cuda:{options.core_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {options.device}')

    place_cells = PlaceCells(options)
    model = RNN(options, place_cells)
    model = model.to(options.device)

    trajectory_generator = TrajectoryGenerator(options, place_cells)
    trainer = Trainer(options, model, trajectory_generator)
    trainer.train(n_epochs=options.n_epochs, n_steps=options.n_steps)
