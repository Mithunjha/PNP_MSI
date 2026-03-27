import random
import torch
import numpy as np
import os
from torch.utils.data import Dataset
import argparse

def IonImg_show(data, target_size, isVal):
    if isVal:
        data = np.nan_to_num(data, nan=0.0)
        mask = np.where(data>0, 1, 0)

        x_ = 0;y_ = 35
        data = data[x_:x_+target_size, y_:y_+target_size]
        mask = mask[x_:x_+target_size, y_:y_+target_size]
        return data, mask
    else:
        data = np.nan_to_num(data, nan=0.0)
        mask = np.where(data>0, 1, 0)
        x, y = data.shape
        x_ = random.randint(0, x-target_size)
        y_ = random.randint(0, y-target_size)
        data = data[x_:x_+target_size, y_:y_+target_size]
        mask = mask[x_:x_+target_size, y_:y_+target_size]
        return data, mask


def IonImg_show_testing(data, target_size):
    
    data = np.nan_to_num(data, nan=0.0)
    mask = np.where(data>0, 1, 0)

    h, w = data.shape  # Get current image size
    
    # Case 1: If the image is smaller than target_size, pad it
    if h < target_size:
        pad_h = max(0, target_size - h)

        data = np.pad(data, ((pad_h // 2, pad_h - pad_h // 2), (0,0)), 
                        mode='constant', constant_values=0)

        mask = np.pad(mask, ((pad_h // 2, pad_h - pad_h // 2), (0,0)), 
                        mode='constant', constant_values=0)
    if w < target_size:
        pad_w = max(0, target_size - w)
        data = np.pad(data, ((0,0), (pad_w // 2, pad_w - pad_w // 2)), 
                      mode='constant', constant_values=0) 
        mask = np.pad(mask, ((0,0), (pad_w // 2, pad_w - pad_w // 2)),
                        mode='constant', constant_values=0)
        

    # Case 2: If the image is larger than target_size, crop it
    if h > target_size:
        x_ = 0 #random.randint(0, h-target_size)
        data = data[x_:x_+target_size, :]
        mask = mask[x_:x_+target_size, :]
    if w > target_size:
        y_ = 0 #random.randint(0, w-target_size)
        data = data[:, y_:y_+target_size]
        mask = mask[:, y_:y_+target_size]
    return data,mask





def read_npy(path, isVal, isTest = False, ROI = None):
    if (isVal==False) and (isTest == False):
        k = 0
        substrings = ["R00","R01", "R02", "R03", "R04"]
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".npy") and any(substring in file for substring in substrings):
                    path = os.path.join(root, file)
                    print(f"Reading from {path} ====================================================")
                    if k == 0:
                        data = np.load(path)
                        print(f"Shape of the data : {data.shape}")
                        k += 1
                    elif k > 0:
                        data = np.concatenate((data, np.load(path)), axis = 0)
                        print(f"Shape of the data : {data.shape}")

        _target = data
        print(f"Number of samples (Training) : {_target.shape[0]}")
        print(f"Shape of each data (Training) : {_target.shape}")

    elif (isVal==True) and (isTest == False):
        k = 0
        substrings = ["R05"]
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".npy") and any(substring in file for substring in substrings):
                    path = os.path.join(root, file)
                    print(f"Reading from {path} ====================================================")
                    if k == 0:
                        data = np.load(path)
                        print(f"Shape of the data : {data.shape}")
                        k += 1
                    elif k > 0:
                        data = np.concatenate((data, np.load(path)), axis = 0)
                        print(f"Shape of the data : {data.shape}")
        _target = data
        print(f"Number of samples (Validation) : {_target.shape[0]}")
        print(f"Shape of each data (Validation) : {_target.shape}")

    if isTest==True:
        k = 0
        substrings = ROI
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".npy") and any(substring in file for substring in substrings):
                    path = os.path.join(root, file)
                    print(f"Reading from {path} ====================================================")
                    if k == 0:
                        data = np.load(path)
                        print(f"Shape of the data : {data.shape}")
                        k += 1
                    elif k > 0:
                        data = np.concatenate((data, np.load(path)), axis = 0)
                        print(f"Shape of the data : {data.shape}")
        _target = data
        print(f"Number of samples (Validation) : {_target.shape[0]}")
        print(f"Shape of each data (Validation) : {_target.shape}")


    return _target

class HDF5Dataset(Dataset):
    def __init__(self,img_dir, isVal, target_size = 192, device = None,noise_sigma = [0.01,.2], add_noise = True, if_seed = False,
    isTest = False, ROI = None):
        self.img_dir = img_dir
        self.ground_truth = read_npy(self.img_dir,isVal = bool(isVal),isTest = bool(isTest), ROI = ROI) 
        self.device = device
        self.target_size = target_size
        self.noise_sigma = noise_sigma
        self.add_noise = add_noise
        self.isVal = isVal
        self.if_seed = if_seed

        print("Number of samples : ", self.ground_truth.shape)
        
    def __len__(self):
        return self.ground_truth.shape[0]

    def __getitem__(self, index):
        target = self.ground_truth[index,:,:]
    
        ion_img_gt,_ = IonImg_show(target, target_size=self.target_size,isVal=self.isVal)
  
        ion_img_gt = torch.from_numpy(np.divide(ion_img_gt,ion_img_gt.max())).float().to(self.device)
        ion_img_in = add_noise_func(ion_img_gt,  target_size = self.target_size, noise_sigma_range= self.noise_sigma,  if_seed = self.if_seed, device = self.device).squeeze()


        return ion_img_gt.unsqueeze(0), ion_img_in.unsqueeze(0)




def add_noise_func(image, noise_sigma_range = [0.01, 0.2],  target_size=192, device=None, if_seed = False, seed = 42):
    if if_seed:
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    noise_sigma = random.uniform(noise_sigma_range[0], noise_sigma_range[1])
    noise = noise_sigma * torch.randn_like(image)
    image_noisy = image + noise
    image_noisy = torch.clamp(image_noisy, 0, 1)
    return image_noisy




