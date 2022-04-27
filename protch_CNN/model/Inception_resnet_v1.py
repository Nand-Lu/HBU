import torch
import torch.nn as nn
from hyper_params  import  hyper_params

class conv3x3(nn.Module):
    def __init__(self, in_planes, out_channels, stride=1, padding=0):
        super(conv3x3, self).__init__()
        self.conv3x3 = nn.Sequential(
             nn.Conv2d(in_planes, out_channels, kernel_size=3, stride=stride, padding=padding),
             nn.BatchNorm2d(out_channels),
             nn.ReLU()
        )


    def forward(self, input):
        return self.conv3x3(input)
class conv1x1(nn.Module):
    def __init__(self, in_planes, out_channels, stride=1, padding=0):
        super(conv1x1, self).__init__()
        self.conv1x1 = nn.Sequential(
             nn.Conv2d(in_planes, out_channels, kernel_size=1, stride=stride, padding=padding),
             nn.BatchNorm2d(out_channels),
             nn.ReLU()
        )


    def forward(self, input):
        return self.conv1x1(input)



class StemV1(nn.Module):
    def __init__(self, in_planes):
        super(StemV1, self).__init__()


        self.conv1 = conv3x3(in_planes =in_planes,out_channels=32,stride=2, padding=0)
        self.conv2 = conv3x3(in_planes=32, out_channels=32,  stride=1, padding=0)
        self.conv3 = conv3x3(in_planes=32, out_channels=64, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3,  stride=2, padding=0)
        self.conv4 = conv3x3(in_planes=64, out_channels=64,  stride=1, padding=1)
        self.conv5 = conv1x1(in_planes =64,out_channels=80,  stride=1, padding=0)
        self.conv6 = conv3x3(in_planes=80, out_channels=192, stride=1, padding=0)
        self.conv7 = conv3x3(in_planes=192, out_channels=256,  stride=2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x

class Inception_ResNet_A(nn.Module):
    def __init__(self, input, scale=0.3):
        super(Inception_ResNet_A, self).__init__()
        self.conv1 = conv1x1(in_planes =input,out_channels=32,stride=1, padding=0)
        self.conv2 = conv3x3(in_planes=32, out_channels=32, stride=1, padding=1)
        self.line =  nn.Conv2d(96, 256, 1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU()

        self.scale = scale
    def forward(self, x):

        c1 = self.conv1(x)
        # print("c1",c1.shape)
        c2 = self.conv1(x)
        # print("c2", c2.shape)
        c3 = self.conv1(x)
        # print("c3", c3.shape)
        c2_1 = self.conv2(c2)
        # print("c2_1", c2_1.shape)
        c3_1 = self.conv2(c3)
        # print("c3_1", c3_1.shape)
        c3_2 = self.conv2(c3_1)
        # print("c3_2", c3_2.shape)
        cat = torch.cat([c1, c2_1, c3_2],dim=1)#torch.Size([4, 96, 15, 15])
        # print("x",x.shape)

        line = self.line(cat)
        # print("line",line.shape)
        out =self.scale*x +line
        out = self.relu(out)

        return out

class Inception_ResNet_B(nn.Module):
    def __init__(self, input, scale=0.3):
        super(Inception_ResNet_B, self).__init__()
        self.conv1 = conv1x1(in_planes =input,out_channels=128,stride=1, padding=0)
        self.conv1x7 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(1,7), padding=(0,3))
        self.conv7x1 = nn.Conv2d(in_channels=128, out_channels=128,kernel_size=(7,1), padding=(3,0))
        self.line = nn.Conv2d(256, 896, 1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU()

        self.scale = scale
    def forward(self, x):

        c1 = self.conv1(x)
        # print("c1",c1.shape)
        c2 = self.conv1(x)
        # print("c2", c2.shape)
        c2_1 = self.conv1x7(c2)
        # print("c2_1", c2_1.shape)

        c2_1 = self.relu(c2_1)

        c2_2 = self.conv7x1(c2_1)
        # print("c2_2", c2_2.shape)

        c2_2 = self.relu(c2_2)

        cat = torch.cat([c1, c2_2], dim=1)
        line = self.line(cat)
        out = self.scale*x+line

        out = self.relu(out)
        # print("out", out.shape)
        return out


class Inception_ResNet_C(nn.Module):
    def __init__(self, input, scale=0.3):
        super(Inception_ResNet_C, self).__init__()
        self.conv1 = conv1x1(in_planes=input, out_channels=192, stride=1, padding=0)
        self.conv1x3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 3), padding=(0,1))
        self.conv3x1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 1), padding=(1,0))
        self.line = nn.Conv2d(384, 1792, 1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU()

        self.scale =scale
    def forward(self, x):
        c1 = self.conv1(x)
        # print("x", x.shape)
        # print("c1",c1.shape)
        c2 = self.conv1(x)
        # print("c2", c2.shape)
        c2_1 = self.conv1x3(c2)
        # print("c2_1", c2_1.shape)

        c2_1 = self.relu(c2_1)
        c2_2 = self.conv3x1(c2_1)
        # print("c2_2", c2_2.shape)

        c2_2 = self.relu(c2_2)
        cat = torch.cat([c1, c2_2], dim=1)
        # print("cat", cat.shape)
        line = self.line(cat)
        out = self.scale *x + line
        # print("out", out.shape)
        out = self.relu(out)

        return out

class Reduction_A(nn.Module):
    def __init__(self, input,n=384,k=192,l=192,m=256):
        super(Reduction_A, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3,  stride=2, padding=0)
        self.conv1 = conv3x3(in_planes=input, out_channels=n,stride=2,padding=0)
        self.conv2 = conv1x1(in_planes=input, out_channels=k,padding=1)
        self.conv3 = conv3x3(in_planes=k,  out_channels=l,padding=0)
        self.conv4 = conv3x3(in_planes=l,  out_channels=m,stride=2,padding=0)

    def forward(self, x):

        c1 = self.maxpool(x)
        # print("c1",c1.shape)
        c2 = self.conv1(x)
        # print("c2", c2.shape)
        c3 = self.conv2(x)
        # print("c3", c3.shape)
        c3_1 = self.conv3(c3)
        # print("c3_1", c3_1.shape)
        c3_2 = self.conv4(c3_1)
        # print("c3_2", c3_2.shape)
        cat = torch.cat([c1, c2,c3_2], dim=1)

        return cat


class Reduction_B(nn.Module):
    def __init__(self, input):
        super(Reduction_B, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = conv1x1(in_planes=input,  out_channels=256, padding=1)
        self.conv2 = conv3x3(in_planes=256,  out_channels=384, stride=2, padding=0)
        self.conv3 = conv3x3(in_planes=256,  out_channels=256,stride=2, padding=0)
        self.conv4 = conv3x3(in_planes=256,  out_channels=256,  padding=1)
        self.conv5 = conv3x3(in_planes=256,  out_channels=256, stride=2, padding=0)
    def forward(self, x):
        c1 = self.maxpool(x)
        # print("c1", c1.shape)
        c2 = self.conv1(x)
        # print("c2", c2.shape)
        c3 = self.conv1(x)
        # print("c3", c3.shape)
        c4 = self.conv1(x)
        # print("c4", c4.shape)
        c2_1 = self.conv2(c2)
        # print("cc2_1", c2_1.shape)
        c3_1 = self.conv3(c3)
        # print("c3_1", c3_1.shape)
        c4_1 = self.conv4(c4)
        # print("c4_1", c4_1.shape)
        c4_2 = self.conv5(c4_1)
        # print("c4_2", c4_2.shape)
        cat = torch.cat([c1, c2_1, c3_1,c4_2], dim=1)
        # print("cat", cat.shape)
        return cat


class Inception_ResNet_v1(nn.Module):
    def __init__(self,classes=2):
        super(Inception_ResNet_v1, self).__init__()
        blocks = []
        blocks.append(StemV1(in_planes=3))
        for i in range(5):
            blocks.append(Inception_ResNet_A(input=256))
        blocks.append(Reduction_A(input=256))
        for i in range(10):
            blocks.append(Inception_ResNet_B(input=896))
        blocks.append(Reduction_B(input=896))
        for i in range(10):
            blocks.append(Inception_ResNet_C(input=1792))

        self.features = nn.Sequential(*blocks)

        self.avepool = nn.AvgPool2d(kernel_size=3)

        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(1792, classes)

    def forward(self,x):
        x = self.features(x)
        # print("x",x.shape)
        x = self.avepool(x)
        # print("avepool", x.shape)
        x = self.dropout(x)
        # print("dropout", x.shape)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return  x

