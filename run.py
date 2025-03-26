import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
from modules.forward_model import generate_sampling_pattern
from modules.plug_and_play import *
from modules.dataloader import *
from modules.utils import *
from modules.eval_metrics import *

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Plug_and_Play MSI')
    parser.add_argument('--sampling_percentage', type=int, default=50, help='sampling percentage (default: 50)')
    parser.add_argument('--experiment_name', type=str, default= 'FT-ICR_simulated', help='experiment name will be the name of the folder')
    parser.add_argument('--model_path', type=str, default= 'C:/Users/ma127/Mithunjha/PNP_MSI/test/best_model.pth.tar', help='path to the folder to store the model checkpoints')
    parser.add_argument('--max_iteration', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--N', type=int, default=192,
                        help='image size (default: 192)')
    parser.add_argument('--noise_level', type=float, default=0.01,
                        help='Noise level (default: 0.01)')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--rho', type=float, default= 1e-3, help='rho value for ADMM (default: 1e-3)') 
    parser.add_argument('--tol', type=float, default= 1e-8, help='tolerance value for ADMM (default: 1e-8)') 
    parser.add_argument('--data_path', type=str, default= 'C:/Users/ma127/Mithunjha/deep_equilibrium_inverse/dataset', help='path to the folder where the test data is stored')
    parser.add_argument('--seed', type=float, default= 42, help='random seed (default: 42)') 
    parser.add_argument('--save_path', type=str, default= 'C:/Users/ma127/Mithunjha/PNP_MSI/', help='path to the project folder')
    parser.add_argument('--ROI', type=str, default= 'R05', help='ROI name')
    parser.add_argument('--id', type=int, default= 21, help='image id')
    args = parser.parse_args()
    
#     return args


# def main():
#     args = parse_option()
    use_cuda = args.use_gpu and torch.cuda.is_available()
    for arg in vars(args):
        print(f"{arg} : {getattr(args,arg)}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")


    project_path = args.save_path + args.experiment_name + '/' + 'PNP_ADMM_rho_' + str(args.rho) + '_noise_' + str(args.noise_level) + '_max_iter_' + str(args.max_iteration) + '_N_' + str(args.N)
    if not os.path.isdir(project_path):
        os.makedirs(project_path)
        print(f"Project directory created at {project_path}")
    else:
        print(f"Project directory available at {project_path}")


    model = torch.load(args.model_path)
    model = model.to(device)

    sampling_pattern = generate_sampling_pattern(args.sampling_percentage,N=args.N, if_seed=True,seed=args.seed,)
    data_R05 = np.load(f'{args.data_path}/processed_data_{args.ROI}.npy')
    test_data_gt, mask = IonImg_show(data_R05[args.id,:,:], target_size=args.N, isVal=True)
    test_data_gt = torch.from_numpy(np.divide(test_data_gt,test_data_gt.max())).float().unsqueeze(0).unsqueeze(1).to(device)
    test_data_gt = test_data_gt.squeeze().cpu().numpy()
    lr_img = A(test_data_gt, sampling_pattern, N=args.N)
    print(lr_img.shape)

    if args.noise_level > 0:
        seed_everything()  # for reproducibility
        lr_img += np.random.normal(0, args.noise_level, lr_img.shape) # add AWGN
        lr_img = np.clip(lr_img, 0, 1)




    bicubic_init = init_img(lr_img, sampling_pattern.astype(bool), N=args.N)
    print(lr_img.shape, bicubic_init.shape,test_data_gt.shape)


    best_recon, reconstruction, err_list, err_list_X, err_list_U, err_list_Z, rho_list, dict_, PSNR_list_x, SSIM_list_x, = admm_inverse_denoising(A_func=A, A_t_func=AT, 
                                                                                                        pattern=sampling_pattern, b=lr_img, test_data_gt = test_data_gt, rhos =args.rho, tol=args.tol,
                                                                                                        experiment_name=args.experiment_name,N=args.N,max_iters=args.max_iteration, denoiser = model, mask=mask, device=device)


    bicubic_init = bicubic_init*mask.squeeze()
    reconstruction = reconstruction*mask.squeeze()
    lr_img_ = lr_img*mask.squeeze()
    # test_data_gt = test_data_gt*mask.squeeze()

    save_fig_path = project_path + f'/figures/{args.ROI}_{args.id}/'
    if not os.path.isdir(save_fig_path):
        os.makedirs(save_fig_path)
        


    iter_num = range(1,len(err_list)+1)
    plt.figure(figsize=(20,5))
    plt.subplot(1,4,1)
    plt.plot(iter_num, err_list, label='l2 error between recon and groundtruth',linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('l2 difference')
    plt.legend()

    plt.subplot(1,4,2)
    plt.plot(iter_num, err_list_X, label='relative difference for X',linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Convergence of X')
    plt.legend()


    plt.subplot(1,4,3)
    plt.plot(iter_num, err_list_U, label='relative difference for U',linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Convergence of U')
    plt.legend()


    plt.subplot(1,4,4)
    plt.plot(iter_num, err_list_Z, label='relative difference for Z',linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Convergence of Z')
    plt.legend()


    plt.savefig(f"{save_fig_path}/convergence_of_XUZ_rho.png", dpi = 300)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(iter_num, PSNR_list_x, label='PSNR (x)',linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('PSNR')
    plt.legend()
    np.savetxt(f"{save_fig_path}/PSNR_array.csv", PSNR_list_x, delimiter=",")


    plt.subplot(1,2,2)
    plt.plot(iter_num, SSIM_list_x, label='SSIM (x)',linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('SSIM')
    plt.legend()
    np.savetxt(f"{save_fig_path}/SSIM_array.csv", SSIM_list_x, delimiter=",")

    plt.savefig(f"{save_fig_path}/convergence_PSNR_SSIM.png", dpi = 300)



    f, ax = plt.subplots(1,4)
            
    ax[0].axis('off')
    ax[0].imshow(lr_img,cmap='hot',vmin=0,vmax=1.0)
    bbox = ax[0].get_tightbbox(f.canvas.get_renderer())
    f.savefig(save_fig_path + 'lr_img.png',bbox_inches=bbox.transformed(f.dpi_scale_trans.inverted()))
    ax[0].set_title('LR Image')


    ax[1].axis('off')
    ax[1].imshow(bicubic_init,cmap='hot',vmin=0,vmax=1.0)
    bbox = ax[1].get_tightbbox(f.canvas.get_renderer())
    f.savefig(save_fig_path + 'bicubic_img.png',bbox_inches=bbox.transformed(f.dpi_scale_trans.inverted()))
    ax[1].set_title('Interpolated Image')



    ax[2].axis('off')
    ax[2].imshow(reconstruction,cmap='hot',vmin=0,vmax=1.0)
    bbox = ax[2].get_tightbbox(f.canvas.get_renderer())
    f.savefig(save_fig_path + 'PNP_admm.png',bbox_inches=bbox.transformed(f.dpi_scale_trans.inverted()))
    ax[2].set_title('PNP Reconstruction')


    ax[3].axis('off')
    ax[3].imshow(test_data_gt.squeeze(),cmap='hot',vmin=0,vmax=1.0)
    bbox = ax[3].get_tightbbox(f.canvas.get_renderer())
    f.savefig(save_fig_path + 'GT.png',bbox_inches=bbox.transformed(f.dpi_scale_trans.inverted()))
    ax[3].set_title('Ground Truth')

    plt.show()



    mae_in = np.mean((np.uint8((lr_img_.clip(0,1).squeeze()*255.0).round()) -  np.uint8((test_data_gt.clip(0,1).squeeze()*255.0).round()))**2)
    PSNR_in = calculate_psnr(np.uint8((lr_img_.clip(0,1).squeeze()*255.0).round()), np.uint8((test_data_gt.clip(0,1).squeeze()*255.0).round()))
    SSIM_in = calculate_ssim(np.uint8((lr_img_.clip(0,1).squeeze()*255.0).round()), np.uint8((test_data_gt.clip(0,1).squeeze()*255.0).round()))
    print(f'PSNR (Input) = {PSNR_in:.5f}, SSIM (Input) = {SSIM_in:.5f}, MSE (Input) = {mae_in:.5f},')


    mae_before = np.mean((np.uint8((bicubic_init.clip(0,1).squeeze()*255.0).round()) -  np.uint8((test_data_gt.clip(0,1).squeeze()*255.0).round()))**2)
    PSNR_before = calculate_psnr(np.uint8((bicubic_init.clip(0,1).squeeze()*255.0).round()), np.uint8((test_data_gt.clip(0,1).squeeze()*255.0).round()))
    SSIM_before = calculate_ssim(np.uint8((bicubic_init.clip(0,1).squeeze()*255.0).round()), np.uint8((test_data_gt.clip(0,1).squeeze()*255.0).round()))
    print(f'PSNR (Cubic Interpolation) = {PSNR_before:.5f}, SSIM (Cubic Interpolation) = {SSIM_before:.5f}, MSE (Cubic Interpolation) = {mae_before:.5f}, ')

    mae_after = np.mean((np.uint8((reconstruction.clip(0,1).squeeze()*255.0).round()) -  np.uint8((test_data_gt.clip(0,1).squeeze()*255.0).round()))**2)
    PSNR_after = calculate_psnr(np.uint8((reconstruction.clip(0,1).squeeze()*255.0).round()), np.uint8((test_data_gt.clip(0,1).squeeze()*255.0).round()))
    SSIM_after = calculate_ssim(np.uint8((reconstruction.clip(0,1).squeeze()*255.0).round()), np.uint8((test_data_gt.clip(0,1).squeeze()*255.0).round()))
    print(f'PSNR (PNP ADMM) = {PSNR_after:.5f}, SSIM (PNP ADMM) = {SSIM_after:.5f}, MSE (PNP ADMM) = {mae_after:.5f}, ')




if __name__ == '__main__':
    main()