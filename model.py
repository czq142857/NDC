import torch
import torch.nn as nn
import torch.nn.functional as F


class resnet_block(nn.Module):
    def __init__(self, ef_dim):
        super(resnet_block, self).__init__()
        self.ef_dim = ef_dim
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output = self.conv_1(input)
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        output = self.conv_2(output)
        output = output+input
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        return output


class CNN_3d_rec7_resnet(nn.Module):

    def __init__(self, out_bool, out_float, is_undc=False):
        super(CNN_3d_rec7_resnet, self).__init__()
        self.ef_dim = 64
        self.out_bool = out_bool
        self.out_float = out_float
        
        self.conv_0 = nn.Conv3d(1, self.ef_dim, 3, stride=1, padding=0, bias=True)

        self.res_1 = resnet_block(self.ef_dim)
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)

        self.res_2 = resnet_block(self.ef_dim)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)

        self.res_3 = resnet_block(self.ef_dim)
        self.res_4 = resnet_block(self.ef_dim)
        self.res_5 = resnet_block(self.ef_dim)
        self.res_6 = resnet_block(self.ef_dim)
        self.res_7 = resnet_block(self.ef_dim)
        self.res_8 = resnet_block(self.ef_dim)

        self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)

        if self.out_bool:
            if is_undc:
                self.conv_out_bool = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)
            else:
                self.conv_out_bool = nn.Conv3d(self.ef_dim, 1, 1, stride=1, padding=0, bias=True)
        if self.out_float:
            self.conv_out_float = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = x

        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.res_1(out)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.res_2(out)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.res_3(out)
        out = self.res_4(out)
        out = self.res_5(out)
        out = self.res_6(out)
        out = self.res_7(out)
        out = self.res_8(out)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        if self.out_bool and self.out_float:
            out_bool = self.conv_out_bool(out)
            out_float = self.conv_out_float(out)
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.conv_out_bool(out)
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.conv_out_float(out)
            return out_float


class CNN_3d_rec15_resnet(nn.Module):

    def __init__(self, out_bool, out_float, is_undc=False):
        super(CNN_3d_rec15_resnet, self).__init__()
        self.ef_dim = 64
        self.out_bool = out_bool
        self.out_float = out_float
        
        self.conv_0 = nn.Conv3d(1, self.ef_dim, 3, stride=1, padding=1, bias=True)

        self.res_1 = resnet_block(self.ef_dim)
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)

        self.res_2 = resnet_block(self.ef_dim)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)

        self.res_3 = resnet_block(self.ef_dim)
        self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)

        self.res_4 = resnet_block(self.ef_dim)
        self.conv_4 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)

        self.res_5 = resnet_block(self.ef_dim)
        self.conv_5 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)

        self.res_6 = resnet_block(self.ef_dim)
        self.conv_6 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)

        self.res_7 = resnet_block(self.ef_dim)
        self.res_8 = resnet_block(self.ef_dim)
        self.res_9 = resnet_block(self.ef_dim)
        self.res_10 = resnet_block(self.ef_dim)

        self.conv_7 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        self.conv_8 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)

        if self.out_bool:
            if is_undc:
                self.conv_out_bool = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)
            else:
                self.conv_out_bool = nn.Conv3d(self.ef_dim, 1, 1, stride=1, padding=0, bias=True)
        if self.out_float:
            self.conv_out_float = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = x

        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.res_1(out)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.res_2(out)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.res_3(out)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.res_4(out)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.res_5(out)
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.res_6(out)
        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.res_7(out)
        out = self.res_8(out)
        out = self.res_9(out)
        out = self.res_10(out)

        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        if self.out_bool and self.out_float:
            out_bool = self.conv_out_bool(out)
            out_float = self.conv_out_float(out)
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.conv_out_bool(out)
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.conv_out_float(out)
            return out_float




class CNN_3d_rec7(nn.Module):

    def __init__(self, out_bool, out_float, is_undc=False):
        super(CNN_3d_rec7, self).__init__()
        self.ef_dim = 64
        self.out_bool = out_bool
        self.out_float = out_float
        
        self.conv_0 = nn.Conv3d(1, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)

        if self.out_bool:
            if is_undc:
                self.conv_out_bool = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)
            else:
                self.conv_out_bool = nn.Conv3d(self.ef_dim, 1, 1, stride=1, padding=0, bias=True)
        if self.out_float:
            self.conv_out_float = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = x

        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        if self.out_bool and self.out_float:
            out_bool = self.conv_out_bool(out)
            out_float = self.conv_out_float(out)
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.conv_out_bool(out)
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.conv_out_float(out)
            return out_float


class CNN_3d_rec15(nn.Module):

    def __init__(self, out_bool, out_float, is_undc=False):
        super(CNN_3d_rec15, self).__init__()
        self.ef_dim = 32
        self.out_bool = out_bool
        self.out_float = out_float
        
        self.conv_0 = nn.Conv3d(1, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.conv_5 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.conv_6 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.conv_7 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        self.conv_8 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)

        if self.out_bool:
            if is_undc:
                self.conv_out_bool = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)
            else:
                self.conv_out_bool = nn.Conv3d(self.ef_dim, 1, 1, stride=1, padding=0, bias=True)
        if self.out_float:
            self.conv_out_float = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = x

        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        
        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        if self.out_bool and self.out_float:
            out_bool = self.conv_out_bool(out)
            out_float = self.conv_out_float(out)
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.conv_out_bool(out)
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.conv_out_float(out)
            return out_float



