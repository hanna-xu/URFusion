import torch
import torch.nn as nn
from utils import *
import torchvision

import torchvision.transforms.functional as F

class Structure_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(48, 32, kernel_size=5, padding=2, bias=True)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=True)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=True)

        self.conv6 = nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=True)
        self.conv7 = nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=True)

        self.conv8 = nn.Conv2d(96, 64, kernel_size=3, padding=1, bias=True)
        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv10 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.avg_pool_2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.sa_conv1_1 = nn.Conv2d(2, 8, 7, padding=3, bias=True)
        self.sa_conv1_2 = nn.Conv2d(8, 1, 3, padding=1, bias=True)
        self.sa_conv2_1 = nn.Conv2d(2, 8, 7, padding=3, bias=True)
        self.sa_conv2_2 = nn.Conv2d(8, 1, 3, padding=1, bias=True)
        self.sa_conv3_1 = nn.Conv2d(2, 8, 7, padding=3, bias=True)
        self.sa_conv3_2 = nn.Conv2d(8, 1, 3, padding=1, bias=True)
        self.sa_conv4_1 = nn.Conv2d(2, 8, 7, padding=3, bias=True)
        self.sa_conv4_2 = nn.Conv2d(8, 1, 3, padding=1, bias=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)


    def forward(self, x):
        out1 = nn.functional.leaky_relu(self.conv1(x))
        out2 = nn.functional.leaky_relu(self.conv2(out1))

        out3 = self.conv3(torch.cat((out1, out2), dim = 1))
        avg_out3 = torch.mean(out3, dim=1, keepdim=True)
        max_out3, _ = torch.max(out3, dim=1, keepdim=True)
        attention3 = self.sa_conv1_1(torch.cat([avg_out3, max_out3], dim=1))
        attention3 = self.sa_conv1_2(nn.functional.leaky_relu(attention3))
        out3= nn.functional.leaky_relu(out3 * self.sigmoid(attention3))

        out3_ds = self.avg_pool_2(out3)
        avg_out3d = torch.mean(out3_ds, dim=1, keepdim=True)
        max_out3d, _ = torch.max(out3_ds, dim=1, keepdim=True)
        attention3d = self.sa_conv2_1(torch.cat([avg_out3d, max_out3d], dim=1))
        attention3d = self.sa_conv2_2(nn.functional.leaky_relu(attention3d))
        out3_ds= nn.functional.leaky_relu(out3_ds * self.sigmoid(attention3d))

        out4 = nn.functional.relu(self.conv4(out3_ds))

        out4_ds = self.avg_pool_2(out4)
        avg_out4d = torch.mean(out4_ds, dim=1, keepdim=True)
        max_out4d, _ = torch.max(out4_ds, dim=1, keepdim=True)
        attention4d = self.sa_conv3_1(torch.cat([avg_out4d, max_out4d], dim=1))
        attention4d = self.sa_conv3_2(nn.functional.leaky_relu(attention4d))
        out4_ds = nn.functional.leaky_relu(out4_ds * self.sigmoid(attention4d))

        out5 = nn.functional.relu(self.conv5(out4_ds))

        out4_us = nn.functional.interpolate(out4, scale_factor=2, mode='bicubic', align_corners=True)
        out5_us = nn.functional.interpolate(out5, scale_factor=4, mode='bicubic', align_corners=True)

        out6 = nn.functional.leaky_relu(self.conv6(out4_us))
        out7 = nn.functional.leaky_relu(self.conv7(out5_us))

        out8 = self.conv8(torch.cat((out6, out7, out3), dim=1))
        avg_out8 = torch.mean(out8, dim=1, keepdim=True)
        max_out8, _ = torch.max(out8, dim=1, keepdim=True)
        attention8 = self.sa_conv4_1(torch.cat([avg_out8, max_out8], dim=1))
        attention8 = self.sa_conv4_2(nn.functional.leaky_relu(attention8))
        out8 = nn.functional.leaky_relu(out8 * self.sigmoid(attention8))

        out9 = nn.functional.leaky_relu(self.conv9(out8))
        out10 = self.conv10(out9)
        feas = nn.functional.leaky_relu(out10)
        return feas


class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding=0, bias=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=0, bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=0, bias=True)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, padding=0, bias=True)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=True)
        self.conv6 = nn.Conv2d(96, 64, kernel_size=3, padding=0, bias=True)
        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, padding=0, bias=True)
        self.conv8 = nn.Conv2d(64, 32, kernel_size=3, padding=0, bias=True)
        self.conv9 = nn.Conv2d(32, 16, kernel_size=3, padding=0, bias=True)
        self.conv10 = nn.Conv2d(16, 8, kernel_size=3, padding=0, bias=True)
        self.conv11 = nn.Conv2d(8, 4, kernel_size=3, padding=0, bias=True)
        self.conv12 = nn.Conv2d(4, 3, kernel_size=3, padding=0, bias=True)

        self.Norm3 = nn.GroupNorm(num_groups=1, num_channels=3, eps=0.0001, affine=False)
        self.Norm16 = nn.GroupNorm(num_groups=1, num_channels=16, eps=0.001, affine=False)
        self.Norm8 = nn.GroupNorm(num_groups=1, num_channels=8, eps=0.001, affine=False)
        self.Norm4 = nn.GroupNorm(num_groups=1, num_channels=4, eps=0.01, affine=False)

        self.Ins_Norm64 = nn.InstanceNorm2d(num_features=64, eps=0.001, affine=False)
        self.Ins_Norm32 = nn.InstanceNorm2d(num_features=32, eps=0.001, affine=False)
        self.Ins_Norm16 = nn.InstanceNorm2d(num_features=16, eps=0.001, affine=False)

        self.tanh = nn.Tanh()

        self.avg_pool_2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.sa_conv1_1 = nn.Conv2d(2, 8, 5, padding=0, bias=True)
        self.sa_conv1_2 = nn.Conv2d(8, 1, 3, padding=0, bias=True)
        self.sa_conv2_1 = nn.Conv2d(2, 8, 5, padding=0, bias=True)
        self.sa_conv2_2 = nn.Conv2d(8, 1, 3, padding=0, bias=True)
        self.sa_conv3_1 = nn.Conv2d(2, 8, 5, padding=0, bias=True)
        self.sa_conv3_2 = nn.Conv2d(8, 1, 3, padding=0, bias=True)

        self.sigmoid = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(48, 48//2), nn.ReLU(inplace=True), nn.Linear(48//2, 48), nn.Sigmoid())
        self.fc2 = nn.Sequential(nn.Linear(96, 96 // 3), nn.ReLU(inplace=True), nn.Linear(96 // 3, 96), nn.Sigmoid())


    def forward(self, x, alpha1, beta1, alpha2, beta2, alpha3, beta3, r1, r2, modulation=False):
        x = nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')
        out1 = nn.functional.leaky_relu(self.conv1(x))

        out2 = nn.functional.pad(out1, (1, 1, 1, 1), mode='reflect')
        out2 = nn.functional.leaky_relu(self.conv2(out2))

        out3 = nn.functional.pad(out2, (1, 1, 1, 1), mode='reflect')
        out3 = nn.functional.leaky_relu(self.conv3(out3))

        out13 = torch.cat((out3, out1), 1)
        avg_out13 = self.avg_pool(out13).squeeze(-1).squeeze(-1)
        max_out13 = self.max_pool(out13).squeeze(-1).squeeze(-1)
        avg_attention13 = self.fc(avg_out13)
        max_attention13 = self.fc(max_out13)
        attention13 = avg_attention13.unsqueeze(2).unsqueeze(3) + max_attention13.unsqueeze(2).unsqueeze(3)
        out13_atten = out13 * self.sigmoid(attention13)

        out4 = nn.functional.pad(out13_atten, (1, 1, 1, 1), mode='reflect')
        out4 = self.conv4(out4)
        avg_out4 = torch.mean(out4, dim=1, keepdim=True)
        max_out4, _ = torch.max(out4, dim=1, keepdim=True)
        input = torch.cat([avg_out4, max_out4], dim=1)
        input = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        attention4 = nn.functional.leaky_relu(self.sa_conv1_1(input))
        attention4 = nn.functional.pad(attention4, (1, 1, 1, 1), mode = 'reflect')
        attention4 = self.sa_conv1_2(attention4)
        out4 = nn.functional.leaky_relu(out4 * self.sigmoid(attention4))

        out4_ds = self.avg_pool_2(out4)
        out5 = nn.functional.pad(out4_ds, (1, 1, 1, 1), mode='reflect')
        out5 = nn.functional.leaky_relu(self.conv5(out5))

        out3_ds = self.avg_pool_2(out3)
        out35 = torch.cat((out5, out3_ds), 1)
        avg_out35 = torch.mean(out35, dim=1, keepdim=True)
        max_out35, _ = torch.max(out35, dim=1, keepdim=True)
        input = torch.cat([avg_out35, max_out35], dim=1)
        input = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        attention35 = nn.functional.leaky_relu(self.sa_conv2_1(input))
        attention35 = nn.functional.pad(attention35, (1, 1, 1, 1), mode='reflect')
        attention35 = self.sa_conv2_2(attention35)
        out35 = nn.functional.leaky_relu(out35 * self.sigmoid(attention35))

        out35_us = nn.functional.interpolate(out35, scale_factor=2, mode='bicubic', align_corners=True)

        out6 = nn.functional.pad(out35_us, (1, 1, 1, 1), mode='reflect')
        out6 = nn.functional.leaky_relu(self.conv6(out6))

        input = torch.cat((out6, out4), dim=1)
        input = nn.functional.pad(input, (1, 1, 1, 1), mode='reflect')
        out7 = self.conv7(input)
        avg_out7 = torch.mean(out7, dim=1, keepdim=True)
        max_out7, _ = torch.max(out7, dim=1, keepdim=True)
        input = torch.cat([avg_out7, max_out7], dim=1)
        input = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        attention7 = nn.functional.leaky_relu(self.sa_conv3_1(input))
        attention7 = nn.functional.pad(attention7, (1, 1, 1, 1), mode='reflect')
        attention7 = self.sa_conv3_2(attention7)
        out7 = nn.functional.leaky_relu(out7 * self.sigmoid(attention7))

        out8 = nn.functional.pad(out7, (1, 1, 1, 1), mode='reflect')
        out8 = self.conv8(out8)
        out8 = nn.functional.leaky_relu(out8)

        out9 = nn.functional.pad(out8, (1, 1, 1, 1), mode='reflect')
        out9 = self.conv9(out9)
        out9 = nn.functional.leaky_relu(out9)

        out10 = nn.functional.pad(out9, (1, 1, 1, 1), mode='reflect')
        out10 = self.conv10(out10)
        out10 = self.Norm8(out10)
        out10 = nn.functional.leaky_relu(out10)

        out11 = nn.functional.pad(out10, (1, 1, 1, 1), mode='reflect')
        out11 = self.conv11(out11)
        out11 = self.Norm4(out11)
        out11 = nn.functional.leaky_relu(out11)


        out12 = nn.functional.pad(out11, (1, 1, 1, 1), mode='reflect')
        out12 = self.conv12(out12)

        result = self.tanh(out12)/2+0.5
        middle = result

        if modulation:
            middle = alpha1.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3]) * out12 \
                     + beta1.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3])

            middle = self.tanh(middle) / 2 + 0.5

            middle = alpha2.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3]) * middle \
                     + beta2.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3])

            r1 = r1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out10.size(2), out10.size(3))
            middle_ycbcr = rgb2ycbcr(middle)
            middle_y = middle_ycbcr[:, 0:1, :, :]
            middle_y_mean = torch.mean(middle_y, [2, 3]).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out10.size(2),
                                                                                            out10.size(3))
            middle_y_adjust = torch.clamp((middle_y - middle_y_mean) * r1 + middle_y_mean, min=0, max=1)
            middle_adjust_contrast = ycbcr2rgb(
                torch.cat((middle_y_adjust, middle_ycbcr[:, 1:2, :, :], middle_ycbcr[:, 2:3, :, :]), 1))

            middle_adjust_contrast = alpha3.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2],
                                                                               out10.shape[3]) * middle_adjust_contrast \
                                     + beta3.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3])

            r2 = r2.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out10.shape[2], out10.shape[3])
            middle_gray = rgb2gray(middle_adjust_contrast)
            middle_r = middle_adjust_contrast[:, 0:1, :, :]
            middle_g = middle_adjust_contrast[:, 1:2, :, :]
            middle_b = middle_adjust_contrast[:, 2:3, :, :]
            mask = 1 - rgb2gray(
                torch.cat((middle_r - middle_gray, middle_g - middle_gray, middle_b - middle_gray), dim=1))
            middle_r2 = middle_r * (1 + r2 * mask) - middle_gray * (r2 * mask)
            middle_g2 = middle_g * (1 + r2 * mask) - middle_gray * (r2 * mask)
            middle_b2 = middle_b * (1 + r2 * mask) - middle_gray * (r2 * mask)

            result = torch.cat((middle_r2, middle_g2, middle_b2), dim=1)
            result = torch.clamp(result, min=0, max=1)

        return result, middle



class A2V_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=0, bias=True)

        self.fcn_r_1 = nn.Linear(in_features=35, out_features=16)
        self.fcn_r1 = nn.Linear(in_features=16, out_features=1)

        self.fcn_r_2 = nn.Linear(in_features=35, out_features=16)
        self.fcn_r2 = nn.Linear(in_features=16, out_features=1)

        self.fcn_a1 = nn.Linear(in_features=32, out_features=1)
        self.fcn_b1 = nn.Linear(in_features=32, out_features=1)

        self.fcn_a2 = nn.Linear(in_features=32, out_features=1)
        self.fcn_b2 = nn.Linear(in_features=32, out_features=1)

        self.fcn_a3 = nn.Linear(in_features=32, out_features=1)
        self.fcn_b3 = nn.Linear(in_features=32, out_features=1)

        self.reflect_pad = nn.ReflectionPad2d(padding=1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        feas = []
        out1 = nn.functional.leaky_relu(self.conv1(self.reflect_pad(x)))

        output_size = (1, 1)
        pooling = nn.AdaptiveMaxPool2d(output_size)

        input1 = torch.cat((x, out1), 1)
        a1 = self.fcn_a1(pooling(out1).squeeze(-1).squeeze(-1)) +1
        b1 = self.fcn_b1(pooling(out1).squeeze(-1).squeeze(-1))

        a2 = self.fcn_a2(pooling(out1).squeeze(-1).squeeze(-1))+1
        b2 = self.fcn_b2(pooling(out1).squeeze(-1).squeeze(-1))

        a3 = self.fcn_a3(pooling(out1).squeeze(-1).squeeze(-1))+1
        b3 = self.fcn_b3(pooling(out1).squeeze(-1).squeeze(-1))

        r_1 =self.fcn_r_1(pooling(input1).squeeze(-1).squeeze(-1))
        r1 = (self.tanh(self.fcn_r1(nn.functional.leaky_relu(r_1))) + 1) * 2

        r_2 = self.fcn_r_2(pooling(input1).squeeze(-1).squeeze(-1))
        r2 = (self.tanh(self.fcn_r2(nn.functional.leaky_relu(r_2))) + 1) * 2

        return a1, b1, a2, b2, a3, b3, r1, r2