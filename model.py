# -*- coding: utf-8 -*-
from typing import Tuple, Union, List, Dict, Any
import torch

class RNN(torch.nn.Module):
    def __init__(self, options:object, place_cells:object):
        super(RNN, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells

        self.vel_sigma = options.vel_sigma
        self.vel_scale = options.vel_scale
        self.hid_sigma = options.hid_sigma
        self.hid_scale = options.hid_scale

        self.device = options.device

        # Input weights
        self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)

        # RNN
        self.vel_stream = torch.nn.Linear(2, self.Ng, bias=False)
        self.hid_stream = torch.nn.Linear(self.Ng, self.Ng, bias=False)
        self.relu = torch.nn.ReLU()

        # Linear read-out weights
        self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)

        self.softmax = torch.nn.Softmax(dim=-1)

        # Initialize with Xavier/Glorot Uniform Distribution
        for name, param in self.named_parameters():
            if 'bias' in name:
                torch.nn.init.zeros_(param)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param, gain=torch.nn.init.calculate_gain('relu'))
            else:
                raise ValueError(f'Parameter name {name} is not recognized; failed to initialize the parameter.')

    def g(self, inputs:Tuple[torch.Tensor,torch.Tensor]):
        '''
        Compute grid cell activations.
        Args:
            inputs: Tuple of velocity input and the initial state for RNN ([sequence_length, batch_size, 2], [batch_size, Ng]).

        Returns: 
            g: Batch of grid cell activations with shape [sequence_length, batch_size, Ng].
        '''
        vt, p0 = inputs
        assert vt.shape[0] == self.sequence_length and vt.shape[2] == 2, vt.shape

        h = self.encoder(p0)
        g = torch.zeros(vt.shape[0], vt.shape[1], self.Ng, device=self.device)
        for i, v in enumerate(vt):
            vp = self.vel_scale * v + self.vel_sigma * torch.randn(v.shape,device=self.device)
            hp = self.hid_scale * h + self.hid_sigma * torch.randn(h.shape,device=self.device)
            vs = self.vel_stream(vp)
            hs = self.hid_stream(hp)
            h = self.relu(vs + hs)
            g[i] = h
        return g

    def predict(self, inputs:Tuple[torch.Tensor,torch.Tensor]):
        '''
        Predict place cell code.
        Args:
            inputs: Tuple of velocity input and the initial state for RNN ([sequence_length, batch_size, 2], [batch_size, Ng]).

        Returns: 
            place_preds: Predicted place cell activations with shape 
                [sequence_length, batch_size, Np].
        '''
        place_preds = self.decoder(self.g(inputs))
        return place_preds


    def compute_loss(self,
                     inputs:Tuple[torch.Tensor,torch.Tensor],
                     pc_outputs:torch.Tensor,
                     pos:torch.Tensor):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Tuple of velocity input and the initial state for RNN ([sequence_length, batch_size, 2], [batch_size, Ng]).
            pc_outputs: Ground truth place cell activations with shape [sequence_length, batch_size, Np].
            pos: Ground truth 2d position with shape [sequence_length, batch_size, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        y = pc_outputs
        preds = self.predict(inputs)
        yhat = self.softmax(preds)
        loss = -(y*torch.log(yhat)).sum(-1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.hid_stream.weight**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err