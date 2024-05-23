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

def get_data_loader(
        rmaps:dict,
        split:Tuple[int,int,int] = [800,50,50],
        batch_sizes: Tuple[int,int,int] = [64,100,100],
        csize: int = 1,
        seed: int = 1234):
    # randomly select data
    rng = np.random.default_rng(seed)
    wtdata = rmaps['wta'][rng.permutation(len(rmaps['wta']))[:np.sum(split)]]
    tgdata = rmaps['j20a'][rng.permutation(len(rmaps['j20a']))[:np.sum(split)]]

    # introduce channel dimension
    assert wtdata.shape[0]%csize==0 & tgdata.shape[0]%csize==0, 'csize has to divide number of samples'
    wtdata = wtdata.reshape(wtdata.shape[0]//csize,csize,*wtdata.shape[1:])
    tgdata = tgdata.reshape(tgdata.shape[0]//csize,csize,*tgdata.shape[1:])

    # train data
    assert split[0]%csize==0, 'train split is not multiple of csize'
    trn_split = split[0]//csize
    trn_data = np.vstack([wtdata[:trn_split],tgdata[:trn_split]])
    trn_labs = np.array([0]*trn_split + [1]*trn_split)

    # val data
    assert split[1]%csize==0, 'val split is not multiple of csize'
    val_split = split[1]//csize
    val_data = np.vstack([wtdata[trn_split:(trn_split+val_split)],tgdata[trn_split:(trn_split+val_split)]])
    val_labs = np.array([0]*val_split + [1]*val_split)

    # test data
    assert split[2]%csize==0, 'test split is not multiple of csize'
    tst_split = split[2]//csize
    tst_data = np.vstack([wtdata[(trn_split+val_split):],tgdata[(trn_split+val_split):]])
    tst_labs = np.array([0]*tst_split + [1]*tst_split)

    assert trn_data.shape[0]==trn_labs.shape[0]
    assert val_data.shape[0]==val_labs.shape[0]
    assert tst_data.shape[0]==tst_labs.shape[0]
    print(trn_data.shape, trn_labs.shape, val_data.shape, val_labs.shape, tst_data.shape, tst_labs.shape)

    trn_dataset = MyDataset(trn_data, trn_labs, transform=torchvision.transforms.Normalize(0.5,0.5))
    val_dataset = MyDataset(val_data, val_labs, transform=torchvision.transforms.Normalize(0.5,0.5))
    tst_dataset = MyDataset(tst_data, tst_labs, transform=torchvision.transforms.Normalize(0.5,0.5))

    trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_sizes[0], shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_sizes[1], shuffle=True, num_workers=4)
    tst_loader = torch.utils.data.DataLoader(tst_dataset, batch_size=batch_sizes[2], shuffle=True, num_workers=4)

    return trn_loader, val_loader, tst_loader

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

    trn_loader, val_loader, tst_loader = get_data_loader(rmaps, split=[800,50,50], batch_sizes=[args.trn_bsize, args.val_bsize, args.tst_bsize], csize=args.csize, seed=args.seed)

    config = ConvNextV2Config(
        num_channels=args.csize, patch_size=args.patch_size, num_stages=4,
        hidden_sizes=[args.hsize,2*args.hsize,4*args.hsize,8*args.hsize],
        depths=[2,2,6,2], hidden_act='gelu', image_size=args.res, num_labels=2)
    model = ConvNextV2ForImageClassification(config)

    model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    trn_loss = []
    val_loss = []
    val_acc_list = []
    for epoch in range(args.epochs):
        model.train()
        trn_loss_epoch = 0.0
        for data in trn_loader:
            inputs, labels = data
            #inputs = inputs.reshape(inputs.shape[0]//args.csize,args.csize,*inputs.shape[1:])
            print(inputs.shape)
            exit()
            optimizer.zero_grad()
            outputs = model(inputs.to(args.device), labels.to(args.device))
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            trn_loss_epoch += loss.item()
        trn_loss.append(trn_loss_epoch)

        if epoch%5==0:
            model.eval()
            val_loss_epoch = 0.0
            val_acc = []
            for data in val_loader:
                inputs, labels = data
                inputs = inputs.unsqueeze(1)
                with torch.no_grad():
                    outputs = model(inputs.to(args.device),labels.to(args.device))
                pred = (F.softmax(outputs.logits,dim=-1)[:,1] > 0.5).to('cpu').to(float)
                val_acc.extend(list((labels.to(float)==pred).numpy()))

                loss = outputs.loss
                val_loss_epoch += loss.item()
            val_loss.append(val_loss_epoch)
            val_acc_list.append(np.mean(val_acc))
            print(f'Epoch {epoch} / Train Loss: {trn_loss_epoch}, Val Loss: {val_loss_epoch}, Val Acc: {np.mean(val_acc)}')

    model.eval()
    tst_acc = []
    for data in tst_loader:
        inputs, labels = data
        inputs = inputs.unsqueeze(1)
        with torch.no_grad():
            outputs = model(inputs.to(args.device),labels.to(args.device))
        pred = (F.softmax(outputs.logits,dim=-1)[:,1] > 0.5).to('cpu').to(float)
        tst_acc.extend(list((labels.to(float)==pred).numpy()))
    print(f'Test Acc: {np.mean(tst_acc)}')

    out_dict = {'trn_loss':np.array(trn_loss), 'val_loss':np.array(val_loss), 'val_acc':np.array(val_acc_list), 'tst_acc':np.mean(tst_acc)}
    np.savez(f'data/classifier/{model_name}/stats.npz', **out_dict)

    model.save_pretrained(f'data/classifier/{model_name}')