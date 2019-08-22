"""GAN implementation using PyTorch

This GAN is used to generate hebarium snapshots

"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

# --------------- MAIN PARAMETERS -------------------
BATCH_SIZE = 10
IMAGE_SIZE = 128
TRAIN_IMAGES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_data')


# --------- Augmentation parameters
ROTATIONS = 40

# ----------------------------------------------------


# -------------- utility functions -------------------

def initialize_weights(m):
    """
    Takes as input a neural network m that will initialize all its weights.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# ----------------------------------------------------


# --------------- Augmentation step ------------------


random_transforms = [transforms.ColorJitter(), transforms.RandomRotation(degrees=ROTATIONS)]
transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                transforms.CenterCrop(IMAGE_SIZE),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomApply(random_transforms, p=0.2),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# ----------------------------------------------------


# ------------------ original images producer --------
train_data = datasets.ImageFolder(TRAIN_IMAGES_PATH, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
                                           
imgs, label = next(iter(train_loader))
imgs = imgs.numpy().transpose(0, 2, 3, 1)

# ----------------------------------------------------


# ------------------- show arbitrary images ----------
# for i in range(10):
#     plt.imshow(imgs[i])
#     plt.show()
# -----------------------------------------------------


# ----------------- GAN implementation ----------------

# --- GAN parameters ---
INPUT_RANDOM_DIM = 128
DECONV_KERNEL_SIZE = 4


#
CONV_KERNEL_SIZE = 4
INTERMEDIATE_CONV_DIM = 756
SMALL_CONV_DIM = 656
LARGE_CONV_DIM = 1024
RELU_SLOPE = 0.2
# ----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(INPUT_RANDOM_DIM, 512, DECONV_KERNEL_SIZE, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, DECONV_KERNEL_SIZE, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, DECONV_KERNEL_SIZE, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, DECONV_KERNEL_SIZE, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 3, DECONV_KERNEL_SIZE, stride=2, padding=1, bias=False),
                nn.Tanh()
                )
        
    def forward(self, z):
        z = z.view(-1, INPUT_RANDOM_DIM, 1, 1)
        img = self.model(z)
        return img

    
Generator_net = Generator().to(device)
Generator_net.apply(initialize_weights)
print(Generator_net)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
                nn.Conv2d(3, 64, 8, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(64, 128, 8, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(128, 256, 8, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(256, 512, 8, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(512, 1, 4, stride=1, padding=0, bias=False),
                nn.Sigmoid()
                )
        
    def forward(self, input):
        logits = self.main(input)
        return torch.sigmoid(logits).view(-1, 1)

   
    
# Creating the discriminator
Discriminator_net = Discriminator().to(device)
Discriminator_net.apply(initialize_weights)
print(Discriminator_net)
# -----------------------------------------------------

# -------- Optimization preparing ---------------------

NUM_EPOCHES = 10
LEARNING_RATE = 0.001
loss_crit = nn.BCELoss()
Discriminator_opt = optim.Adam(Discriminator_net.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
Generator_opt = optim.Adam(Generator_net.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))


real_label = 0.9
fake_label = 0.0
# -----------------------------------------------------


# ------- Optimization run ----------------------------
for epoch in range(NUM_EPOCHES):
    for ii, (real_images, train_labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        Discriminator_net.zero_grad()
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size, 1), real_label, device=device)

        output = Discriminator_net(real_images)
        print("Network output: ", output.shape)
        errD_real = loss_crit(output, labels)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, INPUT_RANDOM_DIM, 1, 1, device=device)
        fake = Generator_net(noise)
        labels.fill_(fake_label)
        output = Discriminator_net(fake.detach())
        errD_fake = loss_crit(output, labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        Discriminator_opt.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        Generator_net.zero_grad()
        labels.fill_(real_label)  # fake labels are real for generator cost
        output = Discriminator_net(fake)
        errG = loss_crit(output, labels)
        errG.backward()
        D_G_z2 = output.mean().item()
        Generator_opt.step()
        
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        if (ii+1) % (len(train_loader)//2) == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch + 1, epochs, ii+1, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    epoch_time.append(time.time()- start)

