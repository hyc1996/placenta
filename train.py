from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from unet import UNet
# from unet_with_dropout import UNet
from BagData import test_dataloader, train_dataloader,val_dataloader
from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
from unetplusplus import UnetPlusPlus
from deeplabv3 import DeepLabV3Plus
from transunet import TransUNet
from plus2 import UNetPP
from monai.losses import DiceLoss
import csv

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
mini_dice_epc=0
def train(epo_num=10, show_vgg_params=False):
    global mini_dice_epc
    vis = visdom.Visdom()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    # fcn_model = FCNs(pretrained_net=vgg_model, n_class=2)
    # fcn_model = fcn_model.to(device)
    # unetplusplusmodel = UnetPlusPlus(num_classes=2, deep_supervision=False).to(device)
    # unetplusplusmodel= UNetPP()
    deeplabv3_model = DeepLabV3Plus(num_classes=2).to(device)
    # unet_model=UNet(in_channels=3,num_classes=2).to(device)
    #transunet_model = TransUNet(256,3,1,8,2048,8,32,2)
    # fcn_model = TransUNet(3,3,2,3,3,9,9,2)


    # criterion = nn.BCELoss().to(device)
    criterion = DiceLoss()


    # optimizer = optim.RMSprop(deeplabv3_model.parameters(), lr=1e-6, weight_decay=1e-8, momentum=0.9)  # 定义优化器
    optimizer = optim.Adam(deeplabv3_model.parameters(),  lr=5e-6)

    all_train_iter_loss = []
    all_val_iter_loss = []
    minst_val_loss=1.0

    # start timing
    prev_time = datetime.now()
    for epo in range(epo_num):

        train_loss = 0
        # unetplusplusmodel.train()
        deeplabv3_model.train()
        # unetplusplusmodel.train()
        for index, (bag, bag_msk,bag_name) in enumerate(train_dataloader):
            # bag.shape is torch.Size([4, 3, 160, 160])
            # bag_msk.shape is torch.Size([4, 2, 160, 160])

            bag = bag.to(device)
            bag_msk = bag_msk.to(device)

            optimizer.zero_grad()
            output = deeplabv3_model(bag)

            output = torch.sigmoid(output['out']) # output.shape is torch.Size([4, 2, 160, 160])
            loss = criterion(output, bag_msk)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss

            optimizer.step()

            output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)
            output_np = np.argmin(output_np, axis=1)
            bag_msk_np = bag_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160)
            bag_msk_np = np.argmin(bag_msk_np, axis=1)

            if np.mod(index, 15) == 0 and index > 0:
                # print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))
                # vis.close()
                vis.images(output_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction'))
                vis.images(bag_msk_np[:, None, :, :], win='train_label', opts=dict(title='label'))
                vis.line(all_train_iter_loss, win='train_iter_loss',opts=dict(title='train iter loss'))

            # plt.subplot(1, 2, 1)
            # plt.imshow(np.squeeze(bag_msk_np[0, ...]), 'gray')
            # plt.subplot(1, 2, 2)
            # plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
            # plt.pause(0.5)


        val_loss = 0
        deeplabv3_model.eval()
        # unetplusplusmodel.eval()
        with torch.no_grad():
            for index, (bag, bag_msk,bag_name) in enumerate(val_dataloader):

                bag = bag.to(device)
                bag_msk = bag_msk.to(device)

                optimizer.zero_grad()
                output = deeplabv3_model(bag)
                output = torch.sigmoid(output['out']) # output.shape is torch.Size([4, 2, 160, 160])
                loss = criterion(output, bag_msk)
                iter_loss = loss.item()
                all_val_iter_loss.append(iter_loss)
                val_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)
                output_np = np.argmin(output_np, axis=1)
                bag_msk_np = bag_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160)
                bag_msk_np = np.argmin(bag_msk_np, axis=1)


                if np.mod(index, 15) == 0:
                    # print(r'valing... Open http://localhost:8097/ to see test result.')
                    # vis.close()
                    vis.images(output_np[:, None, :, :], win='val_pred', opts=dict(title='val prediction'))
                    vis.images(bag_msk_np[:, None, :, :], win='val_label', opts=dict(title='label'))
                    vis.line(all_val_iter_loss, win='val_iter_loss', opts=dict(title='val iter loss'))

                # plt.subplot(1, 2, 1)
                # plt.imshow(np.squeeze(bag_msk_np[0, ...]), 'gray')
                # plt.subplot(1, 2, 2)
                # plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
                # plt.pause(0.5)


        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time
        train_loss = train_loss / len(train_dataloader)
        val_loss = val_loss / len(val_dataloader)
        print('epo = %f,epoch train loss = %f, epoch val loss = %f, %s'%(epo,train_loss, val_loss, time_str))
        row=[epo,train_loss,val_loss,time_str]

        with open('deeplab-result.csv', 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

        if minst_val_loss>val_loss:
            minst_val_loss=val_loss
            torch.save(deeplabv3_model, 'checkpoints/best_deeplabv3_model5.pt')
            mini_dice_epc=epo
            print('minst_val_loss:',minst_val_loss)
    return mini_dice_epc,minst_val_loss

if __name__ == "__main__":

    mini_dice_epc,minst_val_loss=train(epo_num=1000, show_vgg_params=False)
    print(mini_dice_epc,minst_val_loss)
