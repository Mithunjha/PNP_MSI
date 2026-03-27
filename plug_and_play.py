import torch
import numpy as np
import os
import scipy
from modules.eval_metrics import calculate_PSNR_SSIM 

def Update_H(noisy_x,model, device=None):
    model.eval()
    noisy_x = torch.from_numpy(noisy_x).unsqueeze(0).unsqueeze(0).float()
    noisy_x = torch.clamp(noisy_x, 0, 1)
    
    dn_output = model(noisy_x.to(device)) 

    dn_output = dn_output.detach().cpu().numpy().squeeze()
    return dn_output



def relative_l2_error(x1,x2):
    realtive_l2_error = np.linalg.norm(np.matrix.flatten(x1 - x2)) / np.linalg.norm(np.matrix.flatten(x2))
    return realtive_l2_error


def A(x, sampling_pattern, N):
    return x * sampling_pattern

def AT(x, sampling_pattern):
    upsampled_image = np.zeros((x.shape))
    sampling_pattern = sampling_pattern.astype(bool)
    upsampled_image[sampling_pattern] = x[sampling_pattern]
    return upsampled_image


def cg_solve(A_func, A_t_func, sampling_pattern, b, rho, v, N,  tol=1e-6, max_iter=200):#max_iter = 200; 5 was used  in the original code#1e-6
    def matvec_lhs(x):
        x = x.reshape((N,N))
        out = (A_t_func(A_func(x, sampling_pattern, N = N), sampling_pattern) + rho * x).flatten() 
    
        return out
    

    lhs = scipy.sparse.linalg.LinearOperator((N*N, N*N), matvec=matvec_lhs)
    rhs = A_t_func(b,sampling_pattern).flatten() + rho * v.flatten() 
    x, _ = scipy.sparse.linalg.cg(lhs, rhs, atol=tol, maxiter=max_iter) # was tol, now atol
    return x.reshape((N,N))




def admm_inverse_denoising(A_func, A_t_func, pattern, b,  rhos, tol, experiment_name, N,  denoiser, max_iters=50, mask=None, device = None, test_data_gt= None):
    W, H = N, N
    x_old = np.zeros((W,H))
    z_old = np.zeros((W, H))
    u_old = np.zeros((W, H))

    PSNR_list_x=[]
    SSIM_list_x=[]

    err_i = 1e5
    err_list    = []
    err_list_X  = []
    err_list_U  = []
    err_list_Z  = []
    rho_list    = []

    os.makedirs(f"C:/Users/ma127/Mithunjha/Results/{experiment_name}", exist_ok=True)
    with open(f"C:/Users/ma127/Mithunjha/Results/{experiment_name}/iterations.txt",'a') as f:
        f.write(f'rho_init : {rhos}, max_iters : {max_iters},  tol : {tol} \n\n\n')
        
    for k in range(max_iters):

        if test_data_gt is not None:
            calculate_PSNR_SSIM(x_old,test_data_gt,PSNR_list_x,SSIM_list_x,mask)
        else:
            PSNR_list_x.append(0)
            SSIM_list_x.append(0)

        # Step 1: x = prox_∥·∥₂,ρ(v) = cg_solve(A^T A + ρI, A^T b + ρ(z - u))
        v = z_old - u_old
        x = cg_solve(A_func, A_t_func, pattern, b, rhos, v, N=N)


        # Step 2: prox_D,ρ(x + u) = D(x + u, σ² = λ / ρ)
        # v = D(x, N=N)  + u_old
        v = x + u_old

        # z = soft_thresholding(v, kappa = sigma)
        z = Update_H(v, denoiser, device=device) 
        
        # Step 3: u = u + x - z
        # u = u_old + D(x, N=N) - z
        u = u_old + x - z


        recon_tmp=x
        if test_data_gt is not None:
            err = relative_l2_error(recon_tmp,test_data_gt.squeeze())
        else:
            err = relative_l2_error(recon_tmp,x_old)
            
        err_X = relative_l2_error(x,x_old)
        err_U = relative_l2_error(u,u_old)
        err_Z = relative_l2_error(z,z_old)

        err_list.append(err)
        err_list_X.append(err_X)
        err_list_U.append(err_U)
        err_list_Z.append(err_Z)
        rho_list.append(rhos)

        if err < err_i:
            err_i = err
            recon_best = recon_tmp
            best_iteration = k

        
        delta = (np.linalg.norm(x - x_old) + np.linalg.norm(z - z_old) + np.linalg.norm(u - u_old))/np.sqrt(N)

        if delta < tol:
            with open(f"C:/Users/ma127/Mithunjha/Results/{experiment_name}/iterations.txt",'a') as f:
                f.write(f'Converged at iteration {k} with delta {delta} and tol {tol}\n')
            print(f'Converged at iteration {k} with delta {delta} and tol {tol}')
            break

        with open(f"C:/Users/ma127/Mithunjha/Results/{experiment_name}/iterations.txt",'a') as f:

            f.write(f"num_iter : , {k}, err : {err}, err_X : {err_X}, err_U : {err_U}, err_Z : {err_Z} \n")

        # update
        x_old   = x
        u_old   = u
        z_old   = z


    dict_params = {'rho_init': rhos, 'max_iters': max_iters, 
                   'tol': tol,  'best_iteration': best_iteration}
    dict_ = {'err': err_list, 'err_X': err_list_X, 'err_U': err_list_U, 'err_Z': err_list_Z, 'rho': rho_list, 'params': dict_params}
    np.save(f"C:/Users/ma127/Mithunjha/Results/{experiment_name}/err_lists.npy", dict_)

    return recon_best, x_old, err_list, err_list_X, err_list_U, err_list_Z, rho_list, dict_, PSNR_list_x, SSIM_list_x

