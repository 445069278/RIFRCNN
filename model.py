import torch
import torch.nn as nn
import math
import torch.utils.data


#设计参差快
class Residual_block(nn.Module):
    def __init__(self, channels):
        super(Residual_block, self).__init__()
        #第一次卷积 and BN
        self.conv1 = nn.Conv2d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias = False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        #定义AF函数
        self.prelu = nn.PReLU()

        # 第二次卷积,BN
        self.conv2 = nn.Conv2d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias = False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, input_tensor):
        identity_data = input_tensor    #保存输入的特征信息
        output = self.prelu(self.bn1(self.conv1(input_tensor))) #进行第一次参差卷积,BN,并使用PReLu激活
        output = self.bn2(self.conv2(output))   #第二次卷积,BN
        output = torch.add(output, identity_data)   #在激活之前进行identity mapping
        output = self.prelu(output) #激活后得到output

        return  output  #返回output

class MyGeneratorNet(nn.Module):
    def __init__(self):
        super(MyGeneratorNet, self).__init__()



        #代码一,将MS图像进行卷积
        self.encoder1_LowImage = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.PReLU(),
            self.make_layers(Residual_block, 1, 32)
        )

        self.up_conv_LowImage = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=64,
                kernel_size=2,
                stride=2),
            nn.PReLU()
        )

        self.encoder1_HighImage = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.PReLU(),
            self.make_layers(Residual_block, 1, 32)
        )

        self.down_conv_HighImage = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=2,
                stride=2),
            nn.PReLU()
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.PReLU(),
        )

        self.renet4 = self.make_layers(Residual_block, 1, 128)

        self.down_conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=2,
                stride=2),
            nn.PReLU()
        )

        self.restruct1 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.PReLU()
        )

        self.restruct1_rs = self.make_layers(Residual_block, 1, 256)

        self.up_restruct1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2),
            nn.PReLU()
        )

        self.restruct2 = nn.Sequential(
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=1,
                stride=1),
            nn.PReLU(),
            self.make_layers(Residual_block, 1, 256)
        )

        self.up_restruct2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2),
            nn.PReLU()
        )

        self.renet9 = nn.Sequential(
            nn.Conv2d(
                in_channels=160,
                out_channels=80,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.PReLU(),
            self.make_layers(Residual_block, 1, 80)
        )

        self.outImage = nn.Sequential(
            nn.Conv2d(
                in_channels=80,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.Tanh()
        )

    # 链接多少个参差快,本文只需要1次,所以num_of_layer
    def make_layers(self, require_block, num_of_layer, channels):
        layers = []

        for _ in range(num_of_layer):
            layers.append(require_block(channels))

        return nn.Sequential(*layers)

    def forward(self, pan_image, ms_nir):
        #print('------------------------------------------------------------')
        #print('输入一张pan图像[1*512*512]')
        out_pan_0 = self.encoder1_HighImage(pan_image)
        #print('经代码一的参差快:' + str(out_pan_0.size()))
        out_pan = self.down_conv_HighImage(out_pan_0)
        #print('经代码三的下采样:' + str(out_pan.size()))
        #print('------------------------------------------------------------')
        #print('输入一张ms_nir图像[4*128*128]')
        out_ms_nir = self.encoder1_LowImage(ms_nir)
        #print('经代码一的参差快:' + str(out_ms_nir.size()))
        out_ms_nir = self.up_conv_LowImage(out_ms_nir)
        #print('经代码三的上采样:' + str(out_ms_nir.size()))
        #print('------------------------------------------------------------')
        #代码三,叠加
        out_1 = torch.cat((out_ms_nir,out_pan), dim=1)
        #print('经代码三的叠加[(64+64)*256*256]:' + str(out_1.size()))
        #print('------------------------------------------------------------')
        fusion1 = self.encoder4(out_1)
        #print('经代码四卷积:' +str(fusion1.size()))
        fusion2 = self.renet4(fusion1)
        #print('经代码四参差快:' +str(fusion2.size()))
        fusion3 = self.down_conv4(fusion2)
        #print('经代码四下采样:' +str(fusion3.size()))
        #print('------------------------------------------------------------')
        restruct1 = self.restruct1(fusion3)
        #print('经代码五卷积:' +str(restruct1.size()))
        restruct1_rs = self.restruct1_rs(restruct1)
        #print('经代码五参差快:' +str(restruct1_rs.size()))
        up_restruct1 = self.up_restruct1(restruct1_rs)
        #print('经代码五上采样:' +str(up_restruct1.size()))
        #print('------------------------------------------------------------')
        #代码六,叠加
        out_2 = torch.cat((up_restruct1, fusion1, fusion2), dim=1)
        #print('经代码六叠加[(128+128+128)*256*256]:' +str(out_2.size()))
        #print('------------------------------------------------------------')
        restruct2 = self.restruct2(out_2)
        #print('经代码七卷积和参差:' +str(restruct2.size()))
        #print('------------------------------------------------------------')
        up_restruct2 = self.up_restruct2(restruct2)
        #print('经代码八上采样:' +str(up_restruct2.size()))
        #print('------------------------------------------------------------')
        # 代码八,叠加
        out_3 = torch.cat((up_restruct2, out_pan_0), dim=1)
        #print('经代码八叠加:' +str(out_3.size()))
        #print('------------------------------------------------------------')
        outImage = self.renet9(out_3)
        #print('经代码九卷积和参差快:' +str(outImage.size()))
        #print('------------------------------------------------------------')
        outImage = self.outImage(outImage)
        #print('最后卷积为输出结果[4*512,*512]:' +str(outImage.size()))
        #print('------------------------------------------------------------')
        #print('输入一对高低像素的PAN和MS+NIR,得到高像素结果！')
        #print('------------------------------------------------------------')

        return outImage



