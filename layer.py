import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class SeparableConv2D(nn.Layer):
    def __init__(self,in_channel,out_channel,filter_size,padding,stride=1):
        super(SeparableConv2D, self).__init__()
        self.conv1=nn.Conv2D(in_channel,1,filter_size,stride,padding)
        self.conv2=nn.Conv2D(1,out_channel,kernel_size=1,stride=1,padding=0)

    def forward(self,input):
        conv_out=self.conv1(input)
        out=self.conv2(conv_out)
        return out

class Encoder(nn.Layer):
    def __init__(self,in_channel,out_channel):
        super(Encoder, self).__init__()
        self.separa_convs=nn.Sequential(
            SeparableConv2D(in_channel=in_channel, out_channel=out_channel, filter_size=5, padding="same"),
            nn.BatchNorm(out_channel,act="relu"),
            SeparableConv2D(in_channel=out_channel,out_channel=out_channel,filter_size=5,padding="same"),
            nn.BatchNorm(out_channel,act="relu"),
            nn.MaxPool2D(5,2,2),
        )

        self.conv=nn.Conv2D(in_channel,out_channel,1,2,padding="same")

    def forward(self,input):
        precious=input

        y=self.separa_convs(input)
        residual=self.conv(precious)
        y=paddle.add(y,residual)
        y=F.relu(y)
        return y

class Residual(nn.Layer):
    def __init__(self,in_channel,out_channel):
        super(Residual, self).__init__()
        self.conv=nn.Conv2D(in_channel,out_channel,5,1,padding="same")
        self.bn=nn.BatchNorm(out_channel)

    def forward(self,x):
        y=self.conv(x)
        y=self.bn(y)
        y=F.relu(y)
        return y



