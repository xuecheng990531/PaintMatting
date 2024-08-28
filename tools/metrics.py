import numpy as np
unknown_code = 512
import cv2
import os
epsilon = 1e-6
epsilon_sqr = epsilon ** 2
import torch
import torch.nn  as nn
import torch.nn.functional as F


def generate_trimap(alpha, k_size=3, iterations=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv2.dilate(unknown, kernel, iterations=iterations)
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)

def gauss(x, sigma):
    y = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return y


def dgauss(x, sigma):
    y = -x * gauss(x, sigma) / (sigma ** 2)
    return y


def gaussgradient(im, sigma):
    epsilon = 1e-2
    halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(np.int32)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()

    gx = scipy.ndimage.convolve(im, hx, mode='nearest')
    gy = scipy.ndimage.convolve(im, hy, mode='nearest')

    return gx, gy

def compute_mse(pred, alpha, trimap):
    num_pixels = trimap[trimap>=0.4].sum()+trimap[trimap<=0.8].sum()
    loss=((pred - alpha) ** 2).sum()*(trimap==0.502) / num_pixels+ 1e-8
    return loss


# compute the SAD error given a prediction and a ground truth.
#
def compute_mse(pred, alpha, trimap):
    # Convert inputs to PyTorch tensors if they are not already
    
    # Ensure trimap is in the same shape as pred and alpha
    if pred.shape != alpha.shape or pred.shape != trimap.shape:
        raise ValueError("The shapes of pred, alpha, and trimap must be the same.")
    
    # Calculate number of pixels where trimap == 128
    num_pixels = (trimap == 128).float().sum()
    
    # Compute MSE
    mse = ((pred - alpha) ** 2).sum() / num_pixels
    return mse.item()

def compute_sad(pred, alpha):
    
    # Ensure pred and alpha are the same shape
    if pred.shape != alpha.shape:
        raise ValueError("The shapes of pred and alpha must be the same.")
    
    # Compute SAD
    diff = torch.abs(pred - alpha)
    sad = diff.sum() / 1000
    return sad.item()


def fusion_loss(img, alpha_gt, alpha, eps=1e-6):
    smoothloss=nn.SmoothL1Loss()
    # paper loss
    L_alpha = torch.sqrt(torch.pow(alpha_gt - alpha, 2.) + eps).mean()
    gt_msk_img = torch.cat((alpha_gt, alpha_gt, alpha_gt), 1) * img
    alpha_img = torch.cat((alpha, alpha, alpha), 1) * img
    L_color = torch.sqrt(torch.pow(gt_msk_img - alpha_img, 2.) + eps).mean()
    L_smooth=smoothloss(alpha,alpha_gt)
    return L_alpha + L_color+L_smooth


def compute_gradient_loss(pred, target, trimap):

    pred = pred / 255.0
    target = target / 255.0

    pred_x, pred_y = gaussgradient(pred, 1.4)
    target_x, target_y = gaussgradient(target, 1.4)

    pred_amp = np.sqrt(pred_x ** 2 + pred_y ** 2)
    target_amp = np.sqrt(target_x ** 2 + target_y ** 2)

    error_map = (pred_amp - target_amp) ** 2
    loss = np.sum(error_map[trimap == 128])

    return loss / 1000.


def compute_connectivity_error(pred, target, trimap, step=0.1):
    pred = pred / 255.0
    target = target / 255.0
    h, w = pred.shape

    thresh_steps = list(np.arange(0, 1 + step, step))
    l_map = np.ones_like(pred, dtype=np.float) * -1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).astype(np.int)
        target_alpha_thresh = (target >= thresh_steps[i]).astype(np.int)

        omega = getLargestCC(pred_alpha_thresh * target_alpha_thresh).astype(np.int)
        flag = ((l_map == -1) & (omega == 0)).astype(np.int)
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(np.int)
    target_phi = 1 - target_d * (target_d >= 0.15).astype(np.int)
    loss = np.sum(np.abs(pred_phi - target_phi)[trimap == 128])

    return loss / 1000.

class Train_Log():
    def __init__(self, args):
        self.args = args

        self.save_dir = os.path.join(args.saveDir, args.load)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)

        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')

    def save_model(self, model, epoch):

        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, lastest_out_path)

        model_out_path = "{}/model_obj.pth".format(self.save_dir_model)
        torch.save(
            model,
            model_out_path)

    def load_model(self, model):

        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)
        ckpt = torch.load(lastest_out_path)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(lastest_out_path, ckpt['epoch']))

        return start_epoch, model

    def save_log(self, log):
        self.logFile.write(log + '\n')

if __name__ == '__main__':
    import torch
    x=torch.rand(1,1,512,512)
    y=torch.rand(1,1,512,512)
    z=torch.rand(1,1,512,512)

    print(compute_mse(x,y,z))
    print(compute_sad(x,y))
