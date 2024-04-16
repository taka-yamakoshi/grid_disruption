# -*- coding: utf-8 -*-
# mostly copied from https://github.com/ganguli-lab/grid-pattern-formation/blob/master/trainer.py
from typing import Tuple, Union, List, Dict, Any
import os

import torch
import numpy as np


class Trainer(object):
    def __init__(self,
                 options:object,
                 model:torch.nn.Module,
                 trajectory_generator:object,
                 restore:bool=True):
        self.options = options
        self.model = model
        self.trajectory_generator = trajectory_generator
        lr = self.options.learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.loss = []
        self.err = []

        # Set up checkpoints
        self.ckpt_dir = os.path.join(options.save_dir, options.run_ID)
        ckpt_path = os.path.join(self.ckpt_dir, 'most_recent_model.pth')
        if restore and os.path.isdir(self.ckpt_dir) and os.path.isfile(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path))
            print("Restored trained model from {}".format(ckpt_path))
        else:
            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir, exist_ok=True)
            print("Initializing new model from scratch.")
            print("Saving to: {}".format(self.ckpt_dir))

    def train_step(self,
                   inputs:Tuple[torch.Tensor,torch.Tensor],
                   pc_outputs:torch.Tensor,
                   pos:torch.Tensor):
        ''' 
        Train on one batch of trajectories.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        self.model.zero_grad()

        loss, err = self.model.compute_loss(inputs, pc_outputs, pos)

        loss.backward()
        self.optimizer.step()

        return loss.item(), err.item()

    def train(self,
              n_epochs:int=1000,
              n_steps:int=10,
              save:bool=True):
        ''' 
        Train model on simulated trajectories.

        Args:
            n_steps: Number of training steps
            save: If true, save a checkpoint after each epoch.
        '''

        # Construct generator
        gen = self.trajectory_generator.get_generator()

        for epoch_idx in range(n_epochs):
            for step_idx in range(n_steps):
                inputs, pc_outputs, pos = next(gen)
                loss, err = self.train_step(inputs, pc_outputs, pos)
                self.loss.append(loss)
                self.err.append(err)

                print('Epoch: {}/{}. Step {}/{}. Loss: {}. Err: {}cm'.format(
                    epoch_idx, n_epochs, step_idx, n_steps,
                    np.round(loss, 2), np.round(100 * err, 2)))

            if save:
                # Save checkpoint
                ckpt_path = os.path.join(self.ckpt_dir, 'epoch_{}.pth'.format(epoch_idx))
                torch.save(self.model.state_dict(), ckpt_path)
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir,
                                                                 'most_recent_model.pth'))

            # write out loss and error; updated every epoch
            np.save(os.path.join(self.ckpt_dir, 'loss.npy'), np.array(self.loss))
            np.save(os.path.join(self.ckpt_dir, 'err.npy'), np.array(self.err))