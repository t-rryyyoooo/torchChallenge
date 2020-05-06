import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, n=2, use_bn=True):
        super(DoubleConv, self).__init__()

        self.layers = []
        for i in range(1, n + 1):
            if i == 1:
                x = nn.Conv2d(in_channel, out_channel, (3, 3), stride=1, padding=(1, 1), dilation=1)
            else:
                x = nn.Conv2d(out_channel, out_channel, (3, 3), stride=1, padding=(1, 1), dilation=1)

            self.layers.append(x)

            if use_bn:
                self.layers.append(nn.BatchNorm2d(out_channel))
                
            
            self.layers.append(nn.ReLU())

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class CreateConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n=2, use_bn=True, apply_pooling=True):
        super(CreateConvBlock, self).__init__()
        self.apply_pooling = apply_pooling

        self.DoubleConv = DoubleConv(in_channel, out_channel, n=2, use_bn=use_bn)

        if apply_pooling:
            self.maxpool = nn.MaxPool2d((2, 2))
        
    def forward(self, x):
        x = self.DoubleConv(x)
        convResult = x
        if self.apply_pooling:
            x = self.maxpool(x)

        return x, convResult


class CreateUpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, concat_channel,  n=2, use_bn=True):
        super(CreateUpConvBlock, self).__init__()

        x = nn.ConvTranspose2d(in_channel, in_channel, (2, 2), stride=(2, 2), padding=(0, 0), dilation=1)
        self.convTranspose = x

        self.DoubleConv = DoubleConv(in_channel + concat_channel, out_channel, n=2, use_bn=use_bn)

    def forward(self, x1, x2):
        x1 = self.convTranspose(x1)
        x = torch.cat([x2, x1], dim=1)

        x = self.DoubleConv(x)

        return x

class UNetModel(nn.Module):
    def __init__(self, in_channel, nclasses, use_bn=True, use_dropout=True):
        super(UNetModel, self).__init__()
        self.use_dropout = use_dropout

        self.contracts = []
        self.expands = []

        contract = CreateConvBlock(in_channel, 64, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        contract = CreateConvBlock(64, 128, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        contract = CreateConvBlock(128, 256, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        contract = CreateConvBlock(256, 512, n=2, use_bn=use_bn)
        self.contracts.append(contract)

        self.lastContract = CreateConvBlock(512, 1024, n=2, use_bn=use_bn, apply_pooling=False)

        self.contracts = nn.ModuleList(self.contracts)

        if use_dropout:
            self.dropout = nn.Dropout(0.5)

        expand = CreateUpConvBlock(1024, 512, 512,  n=2, use_bn=use_bn)
        self.expands.append(expand)

        expand = CreateUpConvBlock(512, 256, 256,  n=2, use_bn=use_bn)
        self.expands.append(expand)

        expand = CreateUpConvBlock(256, 128, 128, n=2, use_bn=use_bn)
        self.expands.append(expand)
         
        expand = CreateUpConvBlock(128, 64, 64, n=2, use_bn=use_bn)
        self.expands.append(expand)

        self.expands = nn.ModuleList(self.expands)

        self.segmentation = nn.Conv2d(64, nclasses, (1, 1), stride=1, dilation=1, padding=(0, 0))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        convResults = []
        for contract in self.contracts:
            x, convResult = contract(x)
            convResults.append(convResult)

        convResults = convResults[::-1]
        #convResults = nn.ModuleList(convResults)

        x, _ = self.lastContract(x)
        if self.use_dropout:
            x = self.dropout(x)
            
        for expand, convResult in zip(self.expands, convResults):
            x = expand(x, convResult)

        x = self.segmentation(x)
        x = self.softmax(x)

        return x

if __name__ == "__main__":
    model=UNetModel(5 ,3)
    net_shape = [1, 5, 256, 256]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    dummy_img = torch.rand(net_shape).to(device)

    output = model(dummy_img)
    print('output:', output.size())
