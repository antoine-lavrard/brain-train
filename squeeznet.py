"""
Adaptation of squeeznet module from torch hub to implement mixup, using brain-train code as reference. Reference: 
https://pytorch.org/vision/stable/_modules/torchvision/models/squeezenet.html#squeezenet1_1
https://github.com/vgripon/brain-train

"""
from typing import Literal, Optional
import random

import torch
import torch.nn as nn
import torch.nn.init as init


class Fire(nn.Module):
    """
    Change relative to pytorch hub version :
        - do concatenation before activation function instead of after
        - apply manifold mixup in the forward call
        - use leaky relu if needed (original  paper) (using inplace implementation and negative slope parameter value from brain-train)

    """

    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
        use_leaky_relu: bool = False,
    ) -> None:
        """
        Change relative to pytorch hub version :
            - only use one activation function for the 1x1 and 3x3 activations functions
            - added a parameter to implement leaky_relu instead of relu
        """
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand3x3 = nn.Conv2d(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1
        )
        if use_leaky_relu:
            self.squeeze_activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.final_activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:

            self.squeeze_activation = nn.ReLU(inplace=True)
            self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, lbda=None, perm=None) -> torch.Tensor:
        """
        Change relative to version from torch hub :
            - inclusion of lmbda (lambda) and perm coeficients to perform mixup if needed
            - perform activation on both output simultanously instead of separatly, in order to perform mixup.
        Reasoning behind the implementation of mixup :
            According to the squeeznet paper, the reason behind 1x1 filters was to reduce the amount of computation by transforming
            3x3 filters into 1x1 filters. Thus, no difference should be made between them when performing manifold mixup.
        """
        x = self.squeeze_activation(self.squeeze(x))
        y1 = self.expand1x1(x)
        y2 = self.expand3x3(x)
        x = torch.cat([y1, y2], 1)

        if lbda is not None:
            x = lbda * x + (1 - lbda) * x[perm]

        return self.final_activation(x)


class SqueeznetBackbone(nn.Module):
    """
    difference with pytorch_hub :
        - notion of backbone : correspond to the self.features component of torch_hub. The rest is not implemented here.
        - activation : added leaky relu
        - augmentation : add mixup (need to transform the structure of the code to make it compatible)
    """

    def __init__(self, version: Literal["1_0","1_1"] = "1_0", use_leaky_relu: bool = False) -> None:
        """
        change pytorch_hub:
            - only kept backbone part of the network
            - transform the sequential into a list of layer (in order to implement mixup) (ModuleList function : register modules)
            - added the leaky  relu arguments
            - if use leaky relu, will use the same initialization methode than in easy
        """
        super().__init__()
        # define network
        if version == "1_0":
            self.list_module = nn.ModuleList(
                [
                    nn.Conv2d(3, 96, kernel_size=7, stride=2),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True)
                    if use_leaky_relu
                    else nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(96, 16, 64, 64, use_leaky_relu=use_leaky_relu),
                    Fire(128, 16, 64, 64, use_leaky_relu=use_leaky_relu),
                    Fire(128, 32, 128, 128, use_leaky_relu=use_leaky_relu),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(256, 32, 128, 128, use_leaky_relu=use_leaky_relu),
                    Fire(256, 48, 192, 192, use_leaky_relu=use_leaky_relu),
                    Fire(384, 48, 192, 192, use_leaky_relu=use_leaky_relu),
                    Fire(384, 64, 256, 256, use_leaky_relu=use_leaky_relu),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(512, 64, 256, 256, use_leaky_relu=use_leaky_relu),
                ]
            )
            self.number_layer = 9

        elif version == "1_1":
            self.list_module = nn.ModuleList(
                [
                    nn.Conv2d(3, 64, kernel_size=3, stride=2),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True)
                    if use_leaky_relu
                    else nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(64, 16, 64, 64, use_leaky_relu=use_leaky_relu),
                    Fire(128, 16, 64, 64, use_leaky_relu=use_leaky_relu),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(128, 32, 128, 128, use_leaky_relu=use_leaky_relu),
                    Fire(256, 32, 128, 128, use_leaky_relu=use_leaky_relu),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(256, 48, 192, 192, use_leaky_relu=use_leaky_relu),
                    Fire(384, 48, 192, 192, use_leaky_relu=use_leaky_relu),
                    Fire(384, 64, 256, 256, use_leaky_relu=use_leaky_relu),
                    Fire(512, 64, 256, 256, use_leaky_relu=use_leaky_relu),
                ]
            )
            self.number_layer = 9
        else:
            raise ValueError(
                f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected"
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if use_leaky_relu:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="leaky_relu"
                    )
                else:
                    init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        mixup: Literal[None, "mixup", "manifold "] = None,
        lbda: float = None,
        perm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        change :
            - only do forward pass on backbone part of the network
            - change th forward call to include the mixup element
        """
        mixup_layer = -1
        if mixup == "mixup":
            mixup_layer = 0
        elif mixup == "manifold mixup":
            mixup_layer = random.randint(0, self.number_layer)

        if mixup_layer == 0:
            x = lbda * x + (1 - lbda) * x[perm]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        current_layer = 1
        for module in self.list_module:
            if isinstance(module, nn.Conv2d) or isinstance(module, Fire):
                if mixup_layer == current_layer:
                    x = module(x, lbda, perm)
                else:
                    x = module(x)
                current_layer = current_layer + 1
            else:
                x = module(x)

        x = x.mean(dim=list(range(2, len(x.shape))))
        return x


if __name__ == "__main__":
    # launch as script to check that this layer is working
    import torchinfo
    from tqdm import tqdm

    # variables
    min_H = 32
    device = "cuda:0"
    batch_size = 5
    # tests
    model = SqueeznetBackbone().to(device)
    torchinfo.summary(model, (batch_size, 3, min_H, min_H), device=device)

    # test leaky rely
    model = SqueeznetBackbone(use_leaky_relu=True).to(device)
    torchinfo.summary(model, (batch_size, 3, min_H, min_H), device=device)

    # test mixup
    for i in tqdm(range(100), desc="test of the mixup layer"):
        dummy_input = torch.rand((batch_size, 3, min_H, min_H)).to(device)
        out = model(dummy_input, lbda=0.1, perm=torch.randperm(batch_size))

    print("success !")
