import numpy as np
import math
import cv2
# import lpips
import torch
import warnings
import argparse


warnings.filterwarnings('ignore')

def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()



def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())


# loss_fn_vgg = lpips.LPIPS(net='vgg').to('cuda:0')
# def calculate_lpips(reconstruction,test_data_gt,device=None):
    
#     # clip the images to 0,1
#     reconstruction = torch.tensor(reconstruction.clip(0,1)).float().to(device)  
#     test_data_gt = torch.tensor(test_data_gt.clip(0,1)).float().to(device)
#     # normalize to -1,1
#     reconstruction = (reconstruction - 0.5) / 0.5
#     test_data_gt = (test_data_gt - 0.5) / 0.5
#     # calculate the LPIPS   
    
#     loss = loss_fn_vgg(reconstruction, test_data_gt)
#     return loss.item()


def calculate_PSNR_SSIM(reconstruction,test_data_gt,PSNR_list,SSIM_list, mask=None):

    if mask.any() == None:
        reconstruction = reconstruction
        test_data_gt = test_data_gt
    else:
        reconstruction = reconstruction * mask
        test_data_gt = test_data_gt * mask
        
    PSNR = calculate_psnr(np.uint8((reconstruction.clip(0,1).squeeze()*255.0).round()), np.uint8((test_data_gt.clip(0,1).squeeze()*255.0).round()))
    SSIM = calculate_ssim(np.uint8((reconstruction.clip(0,1).squeeze()*255.0).round()), np.uint8((test_data_gt.clip(0,1).squeeze()*255.0).round()))

    PSNR_list.append(PSNR)
    SSIM_list.append(SSIM)