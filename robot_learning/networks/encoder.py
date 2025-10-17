"""
Code reference:
  https://github.com/MishaLaskin/rad/blob/master/encoder.py
"""

import gymnasium.spaces
import torch
import torch.nn as nn

from .utils import CNN, R3M, ResNet18
from robot_learning.utils import RandomShiftsAug

class Encoder(nn.Module):
    def __init__(self, config, ob_space, device):
        super().__init__()

        self.config = config
        self._encoder_type = config.encoder_type
        self._ob_space = ob_space
        self.aug = RandomShiftsAug()

        self.base = nn.ModuleDict()
        encoder_output_dim = 0
        for k, v in ob_space.spaces.items():
            if len(v.shape) in [3, 4]:
                if self._encoder_type == "mlp":
                    self.base[k] = None
                    encoder_output_dim += gymnasium.spaces.flatdim(v)
                elif self._encoder_type == "r3m":
                    r3m = R3M(config)
                    r3m.to(device)
                    self.base[k] = r3m
                    encoder_output_dim += 2048  
                elif self._encoder_type == "resnet18":
                    resnet18 = ResNet18(config)
                    resnet18.to(device)
                    self.base[k] = resnet18
                    encoder_output_dim += 512               
                else:
                    if len(v.shape) == 3:
                        image_dim = v.shape[0]
                    elif len(v.shape) == 4:
                        image_dim = v.shape[0] * v.shape[1]
                    self.base[k] = CNN(config, image_dim)
                    encoder_output_dim += self.base[k].output_dim
            elif len(v.shape) == 1:
                self.base[k] = None
                encoder_output_dim += gymnasium.spaces.flatdim(v)
            else:
                raise ValueError("Check the shape of observation %s (%s)" % (k, v))

        self.output_dim = encoder_output_dim

    def forward(self, ob: dict, detach_conv=False):
        
        encoder_outputs = []
        for k, v in ob.items():
            if self.base[k] is not None:
                if len(v.shape) == 5 or len(v.shape) == 4:
                    # Image
                    v = v.float()
                    if len(v.shape) == 5:
                        v = v.squeeze(1)
                # NHWC -> NCHW
                if v.shape[-1] == 3 and v.shape[1] != 3:
                    v=v.permute(0, 3, 1, 2)  # B,H,W,C -> B,C,H,W

                if isinstance(self.base[k], CNN):
                    if v.max() > 1.0:
                        v = v / 255.0

                if isinstance(self.base[k], ResNet18):
                    v = v / 255.0
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(v.device)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(v.device)
                    v = (v - mean) / std
                
                # image enhancement
                if self.config.image_agmt and len(v.shape) == 4:
                    v = self.aug(v)
                
                # forward
                out = self.base[k](v, detach_conv=detach_conv)
                encoder_outputs.append(out)

            else:
                # other observation (not image)
                encoder_outputs.append(v.flatten(start_dim=1))

        # make sure the output is of shape [B x D]
        encoder_outputs_2d = []
        for out in encoder_outputs:
            if len(out.shape) == 2:
                encoder_outputs_2d.append(out)
            else:
                # flatten the output if it's not 2D
                encoder_outputs_2d.append(out.flatten(start_dim=1))

        out = torch.cat(encoder_outputs_2d, dim=-1)
        assert len(out.shape) == 2
        return out

    def copy_conv_weights_from(self, source):
        """ Tie convolutional layers """
        for k in self.base.keys():
            if self.base[k] is not None:
                self.base[k].copy_conv_weights_from(source.base[k])
