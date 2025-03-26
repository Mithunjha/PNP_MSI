import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
from modules.dataloader import HDF5Dataset
from modules.model import UnetModel
from torch.optim import lr_scheduler as lrs
from modules.utils import AverageMeter

def train(args , model,opt,data_loader,criterion, device, verbose_freq = 100,is_verbose = True):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_losses = AverageMeter()

    for batch_idx, data_loaded in enumerate(data_loader):
        opt.zero_grad()
        ground_truth, data_in = data_loaded
        reconstructions = model(data_in.to(device))
        loss = criterion(reconstructions,ground_truth.to(device))

        
        
        loss.backward()
        opt.step()

        train_losses.update(loss.data.item())

        if is_verbose:
            if ((batch_idx+1) % verbose_freq == 0): #and (i == 0):
                msg = 'Epoch: [{0}/{3}][{1}/{2}]\t' \
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss {train_loss.val:.5f} ({train_loss.avg:.5f})\t'.format(
                        epoch_idx+1, batch_idx,len(data_loader), args.n_epochs , batch_time=batch_time,
                        data_time=data_time, train_loss=train_losses)
                print(msg)

    return train_losses.avg


def validate(model,data_loader,criterion, device):
    model.eval()
    val_losses = AverageMeter()
    with torch.no_grad():
        for batch_idx, data_loaded in enumerate(data_loader):
            ground_truth,data_in  = data_loaded
            reconstructions = model(data_in.to(device))
            loss = criterion(reconstructions,ground_truth.to(device)) 
            val_losses.update(loss.data.item())

    ground_truth,data_in = next(iter(data_loader))

    with torch.no_grad():
      reconstructions = model(data_in)

    return val_losses.avg


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Plug_and_Play MSI')
    parser.add_argument('--experiment_name', type=str, default= 'Denoising', help='experiment name will be the name of the folder')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--N', type=int, default=192,help='input image size (default: 192)')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.3, 
                        help='Learning rate step gamma (default: 0.3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--noise_level_range', type=list, default=[0.01,0.2],
                        help='noise level range (default: [0.01,0.2])')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--data_directory', type=str, default= 'C:/Users/ma127/Mithunjha/deep_equilibrium_inverse/dataset', help='path to the folder where the test data is stored')
    parser.add_argument('--save_path', type=str, default= 'C:/Users/ma127/Mithunjha/PNP_MSI/test', help='path to the project folder')
    
    args = parser.parse_args()
    
    
    use_cuda = args.use_gpu and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    for arg in vars(args):
        print(f"{arg} : {getattr(args,arg)}")
    
    
    train_dataset = HDF5Dataset(args.data_directory,isVal=False, noise_sigma=args.noise_level_range, target_size= args.N, device= device, if_seed = True)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = HDF5Dataset(args.data_directory,isVal=True,noise_sigma=args.noise_level_range, target_size= args.N, device= device,)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)

    model = UnetModel(in_chans=1, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    lr_scheduler = lrs.StepLR(optimizer, step_size=20, gamma=0.1)



    val_best_loss = 10000000
    train_loss_list = []
    val_loss_list = []

    print("Training started")
    for epoch_idx in range(args.epochs):

        print(f"===========================================Training Epoch : [{epoch_idx+1}/{args.epochs}]===============================================================================")

        train_loss = train(args, model,optimizer,dataloader,criterion,device = device,is_verbose=True)
        val_losses = validate(model,test_dataloader,criterion,device = device,)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_losses)
        if (val_losses < val_best_loss):
            val_best_loss = val_losses
            print("Saving Best Model =======================================>")
            torch.save(model, f'{args.save_path}/best_model.pth.tar')

        if (epoch_idx+1%20)==0:
            print("Saving Checkpoint model =======================================>")
            torch.save(model, f'{args.save_path}/best_model_checkpoint.pth.tar')

        lr_scheduler.step()

    plt.figure()
    plt.plot(train_loss_list)
    plt.title("Train Loss")

    plt.figure()
    plt.plot(val_loss_list)
    plt.title("Val Loss")

if __name__ == '__main__':
    main()