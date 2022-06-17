import torch.nn as nn
import torch
from typing import List, Union, TypeVar, Tuple, Optional, Callable, Type, Any
from thop import profile
from time import time

T = TypeVar('T')

class BasicBlock(nn.Module):

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: Union[T, Tuple[T]],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        input_resolution: int = 1000
    ) -> None:
        super(BasicBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.input_resolution = input_resolution
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.conv1 = nn.Conv1d(in_channels=self.in_planes, out_channels=self.out_planes, kernel_size=self.kernel_size, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=self.out_planes, out_channels=self.out_planes, kernel_size=self.kernel_size, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(self.out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.global_avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu(out)

        res = out
        res = self.global_avg(res)

        return out, res

    def flops(self, input_resolution) -> float:

        flops = 0.0
        flops += (2 * self.in_planes * self.kernel_size - 1) * self.out_planes * input_resolution
        flops += (2 * self.out_planes * self.kernel_size - 1) * self.out_planes * (input_resolution // 2)
        flops += self.input_resolution * self.out_planes

        return flops

    def params(self) -> float:

        parameters = 0.0
        parameters += self.in_planes * self.kernel_size * self.out_planes
        parameters += self.out_planes * self.kernel_size * self.out_planes
        parameters += 2 * self.out_planes

        return parameters

class LFCNN(nn.Module):

    def __init__(
        self,
        block: BasicBlock,
        channels: int,
        layers: List[int] = [1, 2, 4, 8, 16, 32, 64, 64, 128],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        input_resolution: int = 1000
    ) -> None:
        super(LFCNN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.input_resolution = input_resolution
        self.channels = channels

        self.channelsID = [['channel' + str(i+1) + '_block' + str(j+1) for j in range(len(layers)-1)]
                            for i in range(self.channels)]

        self.finallayerID = ['fconv1x1' + str(i+1) for i in range(self.channels)]

        for channelID in self.channelsID:
            for i in range(len(channelID)):
                setattr(LFCNN, channelID[i], block(layers[i], layers[i+1], kernel_size=3))

        for finallayerid in self.finallayerID:
            setattr(LFCNN, finallayerid, nn.Conv1d(in_channels=sum(layers[1:]), out_channels=128, kernel_size=1))

    def _forward_imp(self, x: torch.Tensor) -> torch.Tensor:
        channelconcate = []
        for channelID, attr in enumerate(self.channelsID):
            res = x[:, channelID, :]
            res = torch.unsqueeze(res, 1)
            out = []
            for blockID in attr:
                func = getattr(LFCNN, blockID)
                res, res2 = func(res)
                out.append(res2)
            channelconcate.append(out)

        final_out = []
        for i in range(len(channelconcate)):
            channel_out = torch.cat(channelconcate[i], 1)
            func = getattr(LFCNN, self.finallayerID[i])
            channel_out = func(channel_out)
            final_out.append(channel_out)

        final_out = torch.cat(final_out, 1)
        
        return final_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._forward_imp(x)
        return out

    def flops(
        self,
        layers: List[int] = [1, 2, 4, 8, 16, 32, 64, 64, 128],
        input_resolution: int = 1000
    ) -> float:
        flops = 0.0
        for i in range(len(layers)-1):
            block = BasicBlock(layers[i], layers[i+1], kernel_size=3)
            flops += block.flops(input_resolution)
            input_resolution //= 2
            
        flops += (2 * sum(layers[1:]) * 1 - 1) * 128
        flops *= self.channels
        return flops

    def params(
        self,
        layers: List[int] = [1, 2, 4, 8, 16, 32, 64, 64, 128]
    ) -> flops:
        params = 0.0
        for i in range(len(layers)-1):
            block = BasicBlock(layers[i], layers[i+1], kernel_size=3)
            params += block.params()
        
        params += sum(layers[1:]) * 1 * 128
        params *= self.channels
        return params


def sEMGandFMG(
    block: BasicBlock,
    channels: int = 6,
    layers: List[int] = [1, 4, 8, 8, 16, 64, 128, 128, 256],
) -> LFCNN:
    model = LFCNN(block, channels=channels, layers=layers)
    flops = model.flops(layers=layers)
    params = model.params(layers=layers)
    return model, flops, params

def FMG(
    block: BasicBlock,
    channels: int = 3,
    layers: List[int] = [1, 4, 8, 16, 64, 64, 125, 250, 345],
) -> LFCNN:
    model = LFCNN(block, channels=channels, layers=layers)
    flops = model.flops(layers=layers)
    params = model.params(layers=layers)
    return model, flops, params

def sEMG(
    block: BasicBlock,
    channels: int = 3,
    layers: List[int] = [1, 4, 8, 16, 64, 64, 125, 250, 345],
) -> LFCNN:
    model = LFCNN(block, channels=channels, layers=layers)
    flops = model.flops(layers=layers)
    params = model.params(layers=layers)
    return model, flops, params


if __name__ == '__main__':

    print("============sEMG&FMG===============")
    model, flops, params = sEMGandFMG(BasicBlock)
    print(flops)
    print(params)

    print("==============FMG===============")
    model, flops, params = FMG(BasicBlock)
    print(flops)
    print(params)

    print("==============sEMG===============")
    model, flops, params = sEMG(BasicBlock)
    print(flops)
    print(params)