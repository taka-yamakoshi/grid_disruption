import numpy as np
import os
import time
import argparse
from typing import Tuple, Union, List, Dict, Any

import torch
import torch.nn.functional as F
import torchvision

from transformers import ConvNextV2Config, ConvNextV2ForImageClassification

from utils import seed_everything

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, rmaps, labls, transform=None):
        self.rmaps = rmaps
        self.labls = labls
        self.transform = transform

    def __len__(self):
        return len(self.labls)

    def __getitem__(self, idx):
        img = torch.tensor(self.rmaps[idx].astype(np.float32))
        lbl = torch.tensor(self.labls[idx].astype(np.int64))
        return img, lbl

def add_channels(data: np.ndarray,
                 csize: int = 1):
    assert data.shape[0]%csize==0
    return data.reshape(data.shape[0]//csize,csize,*data.shape[1:])

def get_loader(data_wt: np.ndarray,
               data_tg: np.ndarray,
               bsize: int = 100,
               csize: int = 1):
    data_wt = add_channels(data_wt, csize)
    data_tg = add_channels(data_tg, csize)
    data = np.vstack([data_wt,data_tg])
    labs = np.array([0]*len(data_wt) + [1]*len(data_tg))
    dataset = MyDataset(data, labs, transform=torchvision.transforms.Normalize(0.5,0.5,0.5))
    return torch.utils.data.DataLoader(dataset, batch_size=bsize, shuffle=True, num_workers=4)

def get_data(
        rmaps:dict,
        split:Tuple[int,int,int] = [700,100,100],
        seed: int = 1234):
    # randomly select data
    rng = np.random.default_rng(seed)
    wtdata = rmaps['wta'][rng.permutation(len(rmaps['wta']))[:np.sum(split)]]
    tgdata = rmaps['j20a'][rng.permutation(len(rmaps['j20a']))[:np.sum(split)]]

    trn_split = split[0]
    val_split = split[1]
    tst_split = split[2]
    
    trn_data = (wtdata[:trn_split], tgdata[:trn_split])
    val_data = (wtdata[trn_split:(trn_split+val_split)],tgdata[trn_split:(trn_split+val_split)])
    tst_data = (wtdata[(trn_split+val_split):],tgdata[(trn_split+val_split):])
    assert tst_data[0].shape[0]==tst_split and tst_data[1].shape[0]==tst_split
    return trn_data, val_data, tst_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=35)
    parser.add_argument('--sigma', type=int, default=2)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--permute', action='store_true')
    parser.add_argument('--edge', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--trn_bsize', type=int, default=64)
    parser.add_argument('--val_bsize', type=int, default=100)
    parser.add_argument('--tst_bsize', type=int, default=100)
    parser.add_argument('--hsize', type=int, default=20)
    parser.add_argument('--patch_size', type=int, default=3)
    parser.add_argument('--csize', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--core_id', type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)

    args.device = torch.device(f'cuda:{args.core_id}' if torch.cuda.is_available() else 'cpu')

    run_ID = f'{args.res}-{args.sigma}-{args.edge}'
    run_ID = f'{run_ID}-shuffled' if args.shuffle else run_ID
    run_ID = f'{run_ID}-permuted' if args.permute else run_ID
    print(f'Running {run_ID}')

    model_name = f'{run_ID}_{args.trn_bsize}-{args.val_bsize}-{args.tst_bsize}_{args.hsize}-{args.patch_size}-{args.csize}_{args.lr}_{args.seed}'
    os.makedirs(f'data/classifier/{model_name}/',exist_ok=True)

    fname = f'data/rmaps/{run_ID}/rmaps.npz'
    print(f'Loaded {run_ID} --- Last modified at {time.ctime(os.stat(fname).st_mtime)}')
    rmaps = np.load(fname)

    trn_data, val_data, tst_data = get_data(rmaps, split=[700,100,100], seed=args.seed)
    trn_loader = get_loader(trn_data[0], trn_data[1], args.trn_bsize, args.csize)
    val_loader = get_loader(val_data[0], val_data[1], args.val_bsize, args.csize)
    tst_loader = get_loader(tst_data[0], tst_data[1], args.tst_bsize, args.csize)

    config = ConvNextV2Config(
        num_channels=args.csize, patch_size=args.patch_size, num_stages=3,
        hidden_sizes=[args.hsize,2*args.hsize,4*args.hsize],
        depths=[2,2,2], hidden_act='gelu', image_size=args.res, num_labels=2)
    model = ConvNextV2ForImageClassification(config)

    model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    trn_loss = []
    val_loss = []
    val_acc_list = []
    val_loss_best = np.inf
    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed+epoch)
        trn_loader = get_loader(trn_data[0][rng.permutation(len(trn_data[0]))],
                                trn_data[1][rng.permutation(len(trn_data[1]))],
                                args.trn_bsize, args.csize)
        model.train()
        trn_loss_epoch = 0.0
        for data in trn_loader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.to(args.device), labels.to(args.device))
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            trn_loss_epoch += loss.item()
        trn_loss.append(trn_loss_epoch)


        val_loader = get_loader(val_data[0][rng.permutation(len(val_data[0]))],
                                val_data[1][rng.permutation(len(val_data[1]))],
                                args.val_bsize, args.csize)
        model.eval()
        val_loss_epoch = 0.0
        val_acc = []
        for data in val_loader:
            inputs, labels = data
            with torch.no_grad():
                outputs = model(inputs.to(args.device),labels.to(args.device))
            pred = (F.softmax(outputs.logits,dim=-1)[:,1] > 0.5).to('cpu').to(float)
            val_acc.extend(list((labels.to(float)==pred).numpy()))

            loss = outputs.loss
            val_loss_epoch += loss.item()
        if val_loss_epoch < val_loss_best:
            model.save_pretrained(f'data/classifier/{model_name}')
            val_loss_best = val_loss_epoch
        val_loss.append(val_loss_epoch)
        val_acc_list.append(np.mean(val_acc))
        print(f'Epoch {epoch} / Train Loss: {trn_loss_epoch}, Val Loss: {val_loss_epoch}, Val Acc: {np.mean(val_acc)}')

    model = ConvNextV2ForImageClassification.from_pretrained(f'data/classifier/{model_name}')
    model.to(args.device)
    model.eval()
    tst_acc = []
    for data in tst_loader:
        inputs, labels = data
        with torch.no_grad():
            outputs = model(inputs.to(args.device),labels.to(args.device))
        pred = (F.softmax(outputs.logits,dim=-1)[:,1] > 0.5).to('cpu').to(float)
        tst_acc.extend(list((labels.to(float)==pred).numpy()))
    print(f'Test Acc: {np.mean(tst_acc)}')

    out_dict = {'trn_loss':np.array(trn_loss), 'val_loss':np.array(val_loss), 'val_acc':np.array(val_acc_list), 'tst_acc':np.mean(tst_acc)}
    np.savez(f'data/classifier/{model_name}/stats.npz', **out_dict)