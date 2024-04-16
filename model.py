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
        self.input_stream = torch.nn.Linear(2, self.Ng, bias=False)
        self.hiddn_stream = torch.nn.Linear(self.Ng, self.Ng, bias=False)
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
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        '''
        vt, p0 = inputs
        assert vt.shape[1] == self.sequence_length and vt.shape[2] == 2
        assert p0.shape[1] == self.sequence_length and p0.shape[2] == 2

        g = self.encoder(p0)[None]
        for v in torch.unbind(vt, dim=1):
            vp = self.vel_scale * v + self.vel_sigma * torch.randn(v.shape,device=self.device)
            gp = self.hid_scale * g + self.hid_sigma * torch.randn(g.shape,device=self.device)
            i = self.input_stream(vp)
            h = self.hiddn_stream(gp)
            g = self.relu(h + i)
        return g

    def predict(self, inputs:Tuple[torch.Tensor,torch.Tensor]):
        '''
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            place_preds: Predicted place cell activations with shape 
                [batch_size, sequence_length, Np].
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
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        y = pc_outputs
        preds = self.predict(inputs)
        yhat = self.softmax(preds)
        loss = -(y*torch.log(yhat)).sum(-1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.hiddn_stream.weight**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err