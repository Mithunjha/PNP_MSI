import torch
import numpy as np
import random
import os
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
import warnings
from scipy.interpolate import griddata
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

def seed_everything(seed=42):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)

    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x




def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img



def init_img(img,pattern,N):
    img = (img*255).astype(np.uint8)
    sampled_coords = np.array(np.nonzero(pattern)).T
    sampled_values = img[pattern.astype(bool)]
    grid_x, grid_y = np.mgrid[0:N, 0:N]
    X_reconstructed = griddata(sampled_coords, sampled_values, (grid_x, grid_y), method='cubic')
    X_reconstructed[np.isnan(X_reconstructed)] = 0
    return X_reconstructed.astype(np.float64)/255


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def plot_sub_plots(rows,columns,img_list,title_list,cmap = 'hot',vmax = 1, vmin = 0):
    fig, axs = plt.subplots(rows, columns, figsize = (columns*10,rows*10))
    if rows == 1:
        for i in range(len(img_list)):
            im1 = axs[i].imshow(img_list[i],cmap = cmap)#,vmax = vmax, vmin = vmin)
            axs[i].set_title(f"{title_list[i]}")
            plt.colorbar(im1, ax=axs[i],shrink = 0.3)
    else:
        for i in range(len(img_list)):
            im1 = axs[i//columns][i%columns].imshow(img_list[i],cmap = cmap)#,vmax = vmax, vmin = vmin)
            axs[i//columns][i%columns].set_title(f"{title_list[i]}")
            plt.colorbar(im1, ax=axs[i//columns][i%columns],shrink = 0.3)
    # plt.show()
    return fig