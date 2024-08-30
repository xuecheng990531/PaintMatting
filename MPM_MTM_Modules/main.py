import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import os
from model.model730_refine import Unet
from tools.dataset import  matting_datasets
from torchvision.utils import  save_image
from tools.metrics2 import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs")
import numpy as np
import cv2

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='mtm model')
    parser.add_argument('--dataDir', default='datasets/aim500', help='dataset directory')
    parser.add_argument('--saveDir', default='./ckpt', help='model save dir')
    parser.add_argument('--load', default='mtm', help='save model')
    parser.add_argument('--train_batch', type=int, default=12, help='input batch size for train')
    parser.add_argument('--patch_size', type=int, default=512, help='patch size for train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lrDecay', type=int, default=100)
    parser.add_argument('--lrdecayType', default='keep')
    parser.add_argument('--finetuning', default=False)
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--save_epoch', type=int, default=100, help='number of epochs to save model')
    parser.add_argument('--device', type=str, default='cuda', help='cuda device')
    parser.add_argument('--modelname', type=str,default='MTM', help='model name')
    parser.add_argument('--fe', type=str, required=True, help='encoder is frozen')
    parser.add_argument('--norm', type=str,required=True, help='dataset is norm?')

    args = parser.parse_args()
    print(args)
    return args

def main():
    print("=============> Loading args")
    args = get_args()
    device=args.device
    if args.norm=='1':
        norm=True
        str_norm='norm'
    else:
        norm=False
        str_norm='No_norm'

    if args.fe=='1':
        frozen=True
        str_frozen='frozzen'
    else:
        frozen=False
        str_frozen='No_frozzen'

    train_data = matting_datasets(data_root=os.path.join(args.dataDir,'train'),
                                  mode='train',
                                  isnorm=str_norm
                                  )
    trainloader = DataLoader(train_data,
                             batch_size=args.train_batch,
                             shuffle=True,
                             num_workers=0,
                             pin_memory=True)
    test_data = matting_datasets(data_root=os.path.join(args.dataDir,'test'),
                                 mode='test',
                                 isnorm=str_norm
                                 )
    testloader = DataLoader(test_data,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)

    model = Unet(
        backbone_name='resnet18',
        encoder_freeze=frozen
        ).to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr,weight_decay=0.0005,betas=(0.9, 0.999))

    print("============> Start Train ! ...")
    epoch = 0
    loss_ = 0
    while epoch<=args.nEpochs:
        model.train()

        pbar = tqdm(trainloader,desc="Epoch:{}".format(epoch))
        for index, sample_batched in enumerate(pbar):
            image, prompt, alpha, trimap,image_name = sample_batched['image'], sample_batched['prompt'], sample_batched['alpha'], sample_batched['trimap'],sample_batched['image_name']
            image, prompt, alpha, trimap = image.to(device), prompt.to(device), alpha.to(device),trimap.to(device)
            alpha_pre=model(image,prompt)
            if image_name[0]=='o_3c0b2617.jpg' or image_name[0]=='o_fbae5321.jpg' or image_name[0]=='o_f71b4ba5.jpg':
                image_index=image_name[0].split('.')[0]+'.png'
                if os.path.exists(f'pred_alpha_folder/{args.dataDir.split("/")[-1]}/{args.modelname}_{args.lr}_{str_norm}_{str_frozen}'):
                    save_image(torch.cat((alpha_pre,alpha),0), f'pred_alpha_folder/{args.dataDir.split("/")[-1]}/{args.modelname}_{args.lr}_{str_norm}_{str_frozen}/train_{image_index}')
                else:
                    os.makedirs(f'pred_alpha_folder/{args.dataDir.split("/")[-1]}/{args.modelname}_{args.lr}_{str_norm}_{str_frozen}')
                    save_image(torch.cat((alpha_pre,alpha),0), f'pred_alpha_folder/{args.dataDir.split("/")[-1]}/{args.modelname}_{args.lr}_{str_norm}_{str_frozen}/train_{image_index}')

            loss=fusion_loss(image,alpha,alpha_pre)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ += loss.item()

        writer.add_scalar('train loss', loss, epoch)
        writer.close()
        if epoch%2==0:
            print('=============> testing')
            model.eval()
            with torch.no_grad():
                pbar_test = tqdm(testloader,desc='Test')
                for i, sample_batched in enumerate(pbar_test):
                    image, prompt, alpha, trimap,test_image_name = sample_batched['image'], sample_batched['prompt'], sample_batched['alpha'], sample_batched['trimap'],sample_batched['image_name'][0]

                    image, prompt, alpha, trimap = image.to(device), prompt.to(device), alpha.to(device),trimap.to(device)

                    pred_alpha=model(image,prompt)
                    sad, mse, mad=calculate_sad_mse_mad(pred_alpha,alpha,trimap)
                    grad=compute_gradient_whole_image(pred_alpha,alpha)
                    conn=compute_connectivity_loss_whole_image(pred_alpha,alpha)
                    if 'jpg' in test_image_name:
                        test_image_name=test_image_name.replace('jpg','png')
                    if os.path.exists(f'pred_alpha_folder/{args.dataDir.split("/")[-1]}/{args.modelname}_{args.lr}_{str_norm}_{str_frozen}'):
                        save_image(pred_alpha, f'pred_alpha_folder/{args.dataDir.split("/")[-1]}/{args.modelname}_{args.lr}_{str_norm}_{str_frozen}/{test_image_name}')
                    else:
                        os.makedirs(f'pred_alpha_folder/{args.dataDir.split("/")[-1]}/{args.modelname}_{args.lr}_{str_norm}_{str_frozen}')
                        save_image(pred_alpha, f'pred_alpha_folder/{args.dataDir.split("/")[-1]}/{args.modelname}_{args.lr}_{str_norm}_{str_frozen}/{test_image_name}')

                    optimizer.zero_grad()
                    optimizer.step()

                    # Add scalar values to the writer
                    writer.add_scalar('sad', sad, epoch)
                    writer.add_scalar('mse', mse, epoch)
                    writer.add_scalar('sad', conn, epoch)
                    writer.add_scalar('mse', grad, epoch)

                    writer.close()
                    print(
                    'All_model:\n Best MSE:{:.4f}---Best SAD:{:.4f}----Best Grad:{:.4f}---Best Conn:{:.4f}'
                    .format(mse, sad,grad,conn))

        epoch+=1


if __name__ == "__main__":
    main()
