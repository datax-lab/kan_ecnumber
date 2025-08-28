import pandas as pd
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import pickle
from efficient_kan.src.efficient_kan import KAN  

from CLEAN import *
from CLEAN_KAN import CLEAN_KAN


def get_dataloader(dist_map, id_ec, ec_id):
    params = {
        'batch_size': 6000,
        'shuffle': True,
    }
    negative = mine_hard_negative(dist_map, 100)
    train_data = MultiPosNeg_dataset_with_mine_EC(
        id_ec, ec_id, negative, 9, 30)
    return DataLoader(train_data, **params)


def train_model(model, epoch, train_loader, optimizer, device, dtype, criterion):
    model.train()
    total_loss = 0.
    for batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        emb = model(data.to(device=device, dtype=dtype))
        loss = criterion(emb, 0.1, 9)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / (batch + 1)


def val_model(model, epoch, val_loader, device, dtype, criterion):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch, data in enumerate(val_loader):
            emb = model(data.to(device=device, dtype=dtype))
            loss = criterion(emb, 0.1, 9)
            total_loss += loss.item()
    return total_loss / (batch + 1)


def train_config():
    
    # load EC dictionaries
    id_ec, ec_id_dict = get_ec_id_dict('your_training_data.csv')
    val_id_ec, val_ec_id_dict = get_ec_id_dict('your_validation_data.csv')
    ec_id = {k: list(v) for k, v in ec_id_dict.items()}
    val_ec_id = {k: list(v) for k, v in val_ec_id_dict.items()}

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32
    lr, epochs = 5e-4, 1500

    # load pretrained distance maps
    esm_emb = pickle.load(open(
        'your_train_embeddings.pkl','rb')
    ).to(device=device, dtype=dtype)
    dist_map = pickle.load(open(
        'your_train_distance_map.pkl','rb')
    )
    
    val_esm_emb = pickle.load(open(
        'your_val_embeddings.pkl','rb')
    ).to(device=device, dtype=dtype)
    val_dist_map = pickle.load(open(
        'your_val_distance_map.pkl','rb')
    )

    model = CLEAN_KAN()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['Lr'], betas=(0.9, 0.999))
    criterion = SupConHardLoss

    train_loader = get_dataloader(dist_map, id_ec, ec_id)
    val_loader = get_dataloader(val_dist_map, val_id_ec, val_ec_id)
    
    best_vl = float("inf")
    best_path = "models/clean_kan_model.pt"

    for epoch in range(1, epochs + 1):
        # periodically rebuild distance maps
        if epoch % config['Dist'] == 0 and epoch != epochs + 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
            dist_map = get_dist_map(ec_id_dict, esm_emb, device, dtype, model=model)
            train_loader = get_dataloader(dist_map, id_ec, ec_id)
            val_dist_map = get_dist_map(val_ec_id_dict, val_esm_emb, device, dtype, model=model)
            val_loader = get_dataloader(val_dist_map, val_id_ec, val_ec_id)

        tr_loss = train_model(model, epoch, train_loader, optimizer, device, dtype, criterion)
        vl_loss = val_model(model, epoch, val_loader, device, dtype, criterion)
        
        if vl_loss < best_vl:
            best_vl = vl_loss
            torch.save(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": config,
                    "epoch": epoch,
                    "val_loss": vl_loss,
                },
                best_path,
            )

train_config()
