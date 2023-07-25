# This code is based on the code provided in https://github.com/richardkxu/GANs-on-CIFAR10.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        # if args.data == 'mnist' or args.data == 'fashion':
        #     self.in_channels = 1
        #     self.conv1 = nn.Conv2d(1, 196, kernel_size=3, stride=1, padding=1)
        #     size = 28
        # else:
        self.in_channels = 3
        self.conv1 = nn.Conv2d(3, 196, kernel_size=3, stride=1, padding=1)
        size = 128
        
        

        self.ln1 = nn.LayerNorm(normalized_shape=[196, size, size])
        self.lrelu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.ln2 = nn.LayerNorm(normalized_shape=[196, size // 2, size // 2])
        self.lrelu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln3 = nn.LayerNorm(normalized_shape=[196, size // 2, size // 2])
        self.lrelu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.ln4 = nn.LayerNorm(normalized_shape=[196, size // 4, size // 4])
        self.lrelu4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln5 = nn.LayerNorm(normalized_shape=[196, size // 4, size // 4])
        self.lrelu5 = nn.LeakyReLU()

        self.conv6 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln6 = nn.LayerNorm(normalized_shape=[196, size // 4, size // 4])
        self.lrelu6 = nn.LeakyReLU()

        self.conv7 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln7 = nn.LayerNorm(normalized_shape=[196, size // 4, size // 4])
        self.lrelu7 = nn.LeakyReLU()

        self.conv8 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.ln8 = nn.LayerNorm(normalized_shape=[196, size // 8, size // 8])
        self.lrelu8 = nn.LeakyReLU()
        # [B, 196, size // 8, size // 8]

        self.pool = nn.AdaptiveMaxPool2d(1) # global max pool
        # [B, 196, 1, 1]

        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, num_classes)

    def forward(self, x, print_size=False):
        if print_size:
            print("input size: {}".format(x.size()))

        x = self.conv1(x)
        x = self.ln1(x)
        x = self.lrelu1(x)

        if print_size:
            print(x.size())

        x = self.conv2(x)
        x = self.ln2(x)
        x = self.lrelu2(x)

        if print_size:
            print(x.size())

        x = self.conv3(x)
        x = self.ln3(x)
        x = self.lrelu3(x)

        if print_size:
            print(x.size())

        x = self.conv4(x)
        x = self.ln4(x)
        x = self.lrelu4(x)

        if print_size:
            print(x.size())

        x = self.conv5(x)
        x = self.ln5(x)
        x = self.lrelu5(x)

        if print_size:
            print(x.size())

        x = self.conv6(x)
        x = self.ln6(x)
        x = self.lrelu6(x)

        if print_size:
            print(x.size())

        x = self.conv7(x)
        x = self.ln7(x)
        x = self.lrelu7(x)

        if print_size:
            print(x.size())

        x = self.conv8(x)
        x = self.ln8(x)
        x = self.lrelu8(x)

        if print_size:
            print(x.size())

        x = self.pool(x)
        x = x.squeeze()

        if print_size:
            print(x.size())

        x = x.view(x.size(0), -1)

        if print_size:
            print(x.size())

        fc1_out = self.fc1(x)
        fc10_out = self.fc10(x)

        if print_size:
            print("fc1_out size: {}".format(fc1_out.size()))
            print("fc10_out size: {}".format(fc10_out.size()))

        return fc1_out, fc10_out


class Generator(nn.Module):

    def __init__(self, dim_noise):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # in: latent_size x 1 x 1
            nn.ConvTranspose2d(dim_noise, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # out: 512 x 4 x 4
        
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # out: 256 x 8 x 8
        
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # out: 128 x 16 x 16
        
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 64 x 32 x 32

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
                       
            # out: 64 x 64 x 64

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 64 x 128 x 128
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            
        )
        self.dim_noise = dim_noise

    def forward(self, x, print_size=False):
        if print_size:
            print("input size: {}".format(x.size()))

        # x = self.fc1(x)
        # x = self.bn0(x)
        # x = self.relu0(x)

        if print_size:
            print(x.size())

        x = x.view(-1, self.dim_noise, 1, 1)

        if print_size:
            print(x.size())
        
        x = self.gen(x)

        if print_size:
            print("output size: {}".format(x.size()))

        return x


if __name__ == '__main__':
    net1 = Discriminator(10)
    print(net1)
    x = torch.randn(10,3,128,128)
    fc1_out, fc10_out = net1(x, print_size=True)

    net2 = Generator(768)
    print(net2)
    x = torch.randn(10, 768)
    gen_out = net2(x, print_size=True)
