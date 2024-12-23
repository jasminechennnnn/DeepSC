# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:59:14 2020

@author: HQ Xie
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi, plot_training_history
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
import pytz

parser = argparse.ArgumentParser()
#parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='data/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)   #############
parser.add_argument('--dff', default=512, type=int)       #############
parser.add_argument('--num-layers', default=4, type=int)  #############
parser.add_argument('--num-heads', default=8, type=int)   #############
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lamb', default=0.0009, type=float, help='weight for MI loss')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, net):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            loss = val_step(net, sents, sents, 0.1, pad_idx,
                             criterion, args.channel)

            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch, loss
                )
            )

    return total/len(test_iterator)


def train(epoch, args, net, mi_net=None):
    train_eur= EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)

    # for loss history
    total_loss = 0
    total_mi = 0 if mi_net is not None else None
    batch_count = 0

    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))

    for sents in pbar:
        sents = sents.to(device)
        batch_count += 1

        if mi_net is not None:
            mi = train_mi(net, mi_net, sents, 0.1, pad_idx, mi_opt, args.channel)
            loss = train_step(net, sents, sents, 0.1, pad_idx,
                              optimizer, criterion, args.channel, args.lamb, mi_net) # loss = E2E loss + mi
            
            # for loss history 
            total_loss += loss
            total_mi += mi

            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}'.format(
                    epoch, loss, -mi
                )
            )
        else:
            loss = train_step(net, sents, sents, noise_std[0], pad_idx,
                              optimizer, criterion, args.channel)
            
            # for loss history
            total_loss += loss

            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                    epoch, loss
                )
            )
    
    if mi_net is not None:
        return total_loss / batch_count, total_mi / batch_count

    return total_loss / batch_count

if __name__ == '__main__':
    setup_seed(10)
    args = parser.parse_args()
    tw = pytz.timezone('Asia/Taipei')
    current_time = datetime.now(tw)

    args.checkpoint_path = args.checkpoint_path + current_time.strftime('%Y%m%d_%H%M') + '-' + args.channel + '-lamb' + str(args.lamb)
    args.vocab_file = '' + args.vocab_file

    log_dir = os.path.join(args.checkpoint_path, 'log/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, 'train.info.log')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('Training Arguments:\n')
        f.write('-'*50 + '\n')

        for arg in vars(args):
            arg_value = getattr(args, arg)
            f.write(f'{arg}: {arg_value}\n')
        f.write('='*50 + '\n')
        
        f.write(f'Training start time: {current_time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('='*50 + '\n\n')

    print('\n' + '='*50)
    print('Training Arguments:')
    print('-'*50)
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('='*50 + '\n')

    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]


    """ define optimizer and loss function """
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay = 5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    #opt = NoamOpt(args.d_model, 1, 4000, optimizer)
    initNetParams(deepsc)

    train_loss_epochs = []
    train_mi_epochs = []
    val_loss_epochs = []
    min_epoch = 1
    start_epoch = 1
    best_val_loss = 10
    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()

        # val_loss = train(epoch, args, deepsc)
        avg_loss, avg_mi = train(epoch, args, deepsc, mi_net)
        val_loss = validate(epoch, args, deepsc)

        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
        with open(args.checkpoint_path + '/latest.pth'.format(str(epoch).zfill(2)), 'wb') as f:
            torch.save(deepsc.state_dict(), f)

        # only save current best weight
        if val_loss < best_val_loss:
            min_epoch = epoch
            with open(args.checkpoint_path + '/best.pth'.format(str(epoch).zfill(2)), 'wb') as f:
                torch.save(deepsc.state_dict(), f)
            best_val_loss = val_loss
        
        print(f'Current epoch: {epoch}, Best epoch so far: {min_epoch}')

        train_loss_epochs.append(avg_loss)
        train_mi_epochs.append(-avg_mi) #############
        val_loss_epochs.append(val_loss)
        loss_path = args.checkpoint_path + '/losses/'
        os.makedirs(loss_path, exist_ok=True)
        
        np.save(loss_path + 'train_loss_epochs.npy', np.array(train_loss_epochs))
        np.save(loss_path + 'train_mi_epochs.npy', np.array(train_mi_epochs))
        np.save(loss_path + 'vali_loss_epochs.npy', np.array(val_loss_epochs))

        plot_training_history(loss_path, args)

    best_val_loss = []