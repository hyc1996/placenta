import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import cv2
from onehot import onehot
from PIL import Image
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class BagDataset(Dataset):

    def __init__(self, transform=None,m='train'):
        self.transform = transform
        self.m=m
    def __len__(self):
        if self.m== 'train':
            return len(os.listdir('finaljpg256/injpg/train/image'))
        elif self.m== 'val':
            return len(os.listdir('finaljpg256/injpg/val/image'))
        elif self.m == 'in_test':
            return len(os.listdir('finaljpg256/injpg/test/image'))
        elif self.m == 'out_test':
            return len(os.listdir('finaljpg256/outjpg/image'))

    def __getitem__(self, idx):
        if self.m == 'train':
            img_name = os.listdir('finaljpg256/injpg/train/image')[idx]
            imgA = Image.open('finaljpg256/injpg/train/image/'+img_name).convert('RGB')
            imgA = cv2.resize(np.asarray(imgA), (256, 256))
            imgB = Image.open('finaljpg256/injpg/train/label/'+img_name)
            imgB = cv2.resize(np.asarray(imgB), (256, 256))
            imgB = imgB/255
            imgB[imgB>=0.5]=1.0
            imgB[imgB<0.5]=0.0
            imgB = imgB.astype('uint8')
            imgB = onehot(imgB, 2)
            imgB = imgB.transpose(2,0,1)
            imgB = torch.FloatTensor(imgB)
            if self.transform:
                imgA = self.transform(imgA)
            return imgA, imgB,img_name
        elif self.m == 'val':
            img_name = os.listdir('finaljpg256/injpg/val/image')[idx]
            imgA = Image.open('finaljpg256/injpg/val/image/' + img_name).convert('RGB')
            imgA = cv2.resize(np.asarray(imgA), (256, 256))
            imgB = Image.open('finaljpg256/injpg/val/label/' + img_name)
            imgB = cv2.resize(np.asarray(imgB), (256, 256))
            imgB = imgB / 255
            imgB[imgB>=0.5]=1.0
            imgB[imgB<0.5]=0.0
            imgB = imgB.astype('uint8')
            imgB = onehot(imgB, 2)
            imgB = imgB.transpose(2, 0, 1)
            imgB = torch.FloatTensor(imgB)
            if self.transform:
                imgA = self.transform(imgA)
            return imgA, imgB, img_name
        elif self.m == 'in_test':
            img_name = os.listdir('finaljpg256/injpg/test/image')[idx]
            imgA = Image.open('finaljpg256/injpg/test/image/' + img_name).convert('RGB')
            imgA = cv2.resize(np.asarray(imgA), (256, 256))
            imgB = Image.open('finaljpg256/injpg/test/label/' + img_name)
            imgB = cv2.resize(np.asarray(imgB), (256, 256))
            imgB = imgB / 255
            imgB[imgB>=0.5]=1.0
            imgB[imgB<0.5]=0.0
            imgB = imgB.astype('uint8')
            imgB = onehot(imgB, 2)
            imgB = imgB.transpose(2, 0, 1)
            imgB = torch.FloatTensor(imgB)
            if self.transform:
                imgA = self.transform(imgA)
            return imgA, imgB, img_name
        elif self.m == 'out_test':
            img_name = os.listdir('finaljpg256/outjpg/image')[idx]
            imgA = Image.open('finaljpg256/outjpg/image/' + img_name).convert('RGB')
            imgA = cv2.resize(np.asarray(imgA), (256, 256))
            imgB = Image.open('finaljpg256/outjpg/label/' + img_name)
            imgB = cv2.resize(np.asarray(imgB), (256, 256))
            imgB = imgB / 255
            imgB[imgB>=0.5]=1.0
            imgB[imgB<0.5]=0.0
            imgB = imgB.astype('uint8')
            imgB = onehot(imgB, 2)
            imgB = imgB.transpose(2, 0, 1)
            imgB = torch.FloatTensor(imgB)
            if self.transform:
                imgA = self.transform(imgA)
            return imgA, imgB, img_name

train_bag = BagDataset(transform,'train')
test_bag = BagDataset(transform,'in_test')
val_bag = BagDataset(transform,'val')
out_test_bag = BagDataset(transform,'out_test')

train_dataloader = DataLoader(train_bag, batch_size=8, shuffle=True, num_workers=4,drop_last=True)
test_dataloader = DataLoader(test_bag, batch_size=8, shuffle=True, num_workers=4,drop_last=True)
val_dataloader = DataLoader(val_bag, batch_size=8, shuffle=True, num_workers=4,drop_last=True)
out_dataloader = DataLoader(out_test_bag, batch_size=4, shuffle=True, num_workers=4,drop_last=True)



if __name__ =='__main__':
    pass
    # print(len(train_bag),len(test_bag),len(val_bag))

    # for train_batch in train_dataloader:
    #     print(len(train_batch))
    #
    # for test_batch in test_dataloader:
    #     print(len(test_batch))

    #
    # for val_batch in val_dataloader:
    #     print(len(val_batch))