import numpy as np
import torch
from torch import nn
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops


def fusion_loss(img, alpha_gt, alpha, eps=1e-6):
    smoothloss=nn.SmoothL1Loss()
    # paper loss
    L_alpha = torch.sqrt(torch.pow(alpha_gt - alpha, 2.) + eps).mean()
    gt_msk_img = torch.cat((alpha_gt, alpha_gt, alpha_gt), 1) * img
    alpha_img = torch.cat((alpha, alpha, alpha), 1) * img
    L_color = torch.sqrt(torch.pow(gt_msk_img - alpha_img, 2.) + eps).mean()
    L_smooth=smoothloss(alpha,alpha_gt)
    return L_alpha + L_color+L_smooth

def calculate_sad_mse_mad(predict_old, alpha, trimap):
    # Ensure the input tensors are on the CPU and convert them to NumPy arrays
    predict_old_np = predict_old.cpu().detach().numpy()
    alpha_np = alpha.cpu().detach().numpy()
    trimap_np = trimap.cpu().detach().numpy()
    
    batch_size = predict_old_np.shape[0]
    sad_diff_total, mse_diff_total, mad_diff_total = 0, 0, 0
    
    for i in range(batch_size):
        predict = np.copy(predict_old_np[i, 0, :, :])
        alpha_i = alpha_np[i, 0, :, :]
        trimap_i = trimap_np[i, 0, :, :]
        
        pixel = float((trimap_i == 128).sum())
        predict[trimap_i == 255] = 1.
        predict[trimap_i == 0] = 0.
        
        sad_diff = np.sum(np.abs(predict - alpha_i)) / 1000
        if pixel == 0:
            pixel = trimap_i.shape[0] * trimap_i.shape[1] - float((trimap_i == 255).sum()) - float((trimap_i == 0).sum())
        mse_diff = np.sum((predict - alpha_i) ** 2) / pixel
        mad_diff = np.sum(np.abs(predict - alpha_i)) / pixel
        
        sad_diff_total += sad_diff
        mse_diff_total += mse_diff
        mad_diff_total += mad_diff
    
    # Calculate the average over the batch
    sad_diff_avg = sad_diff_total / batch_size
    mse_diff_avg = mse_diff_total / batch_size
    mad_diff_avg = mad_diff_total / batch_size

    return sad_diff_avg, mse_diff_avg, mad_diff_avg



def compute_gradient_whole_image(pd, gt):
    batch_size = pd.shape[0]
    loss_total = 0
    
    pd_np = pd.cpu().detach().numpy()
    gt_np = gt.cpu().detach().numpy()
    
    for i in range(batch_size):
        pd_i = pd_np[i, 0, :, :]
        gt_i = gt_np[i, 0, :, :]
        
        pd_x = gaussian_filter(pd_i, sigma=1.4, order=[1, 0], output=np.float32)
        pd_y = gaussian_filter(pd_i, sigma=1.4, order=[0, 1], output=np.float32)
        gt_x = gaussian_filter(gt_i, sigma=1.4, order=[1, 0], output=np.float32)
        gt_y = gaussian_filter(gt_i, sigma=1.4, order=[0, 1], output=np.float32)
        
        pd_mag = np.sqrt(pd_x**2 + pd_y**2)
        gt_mag = np.sqrt(gt_x**2 + gt_y**2)
        
        error_map = np.square(pd_mag - gt_mag)
        loss = np.sum(error_map) / 10
        loss_total += loss
        
    loss_avg = loss_total / batch_size
    return loss_avg



def compute_connectivity_loss_whole_image(pd, gt, step=0.1):
    batch_size = pd.shape[0]
    loss_total = 0
    
    pd_np = pd.cpu().detach().numpy()
    gt_np = gt.cpu().detach().numpy()
    
    for i in range(batch_size):
        pd_i = pd_np[i, 0, :, :]
        gt_i = gt_np[i, 0, :, :]
        
        h, w = pd_i.shape
        thresh_steps = np.arange(0, 1.1, step)
        l_map = -1 * np.ones((h, w), dtype=np.float32)
        lambda_map = np.ones((h, w), dtype=np.float32)
        
        for j in range(1, thresh_steps.size):
            pd_th = pd_i >= thresh_steps[j]
            gt_th = gt_i >= thresh_steps[j]
            label_image = label(pd_th & gt_th, connectivity=1)
            cc = regionprops(label_image)
            size_vec = np.array([c.area for c in cc])
            if len(size_vec) == 0:
                continue
            max_id = np.argmax(size_vec)
            coords = cc[max_id].coords
            omega = np.zeros((h, w), dtype=np.float32)
            omega[coords[:, 0], coords[:, 1]] = 1
            flag = (l_map == -1) & (omega == 0)
            l_map[flag == 1] = thresh_steps[j-1]
            dist_maps = distance_transform_edt(omega == 0)
            dist_maps = dist_maps / dist_maps.max()
        
        l_map[l_map == -1] = 1
        d_pd = pd_i - l_map
        d_gt = gt_i - l_map
        phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
        phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
        loss = np.sum(np.abs(phi_pd - phi_gt)) / 1000
        loss_total += loss
        
    loss_avg = loss_total / batch_size
    return loss_avg


if __name__=='__main__':
    pd = torch.randn(2, 1, 256, 256)
    gt = torch.randn(2, 1, 256, 256)
    loss = compute_connectivity_loss_whole_image(pd, gt)
    print(f"Connectivity Loss: {loss:.4f}")