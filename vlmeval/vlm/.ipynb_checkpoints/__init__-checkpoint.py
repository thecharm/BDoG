import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)
from .instructblip import InstructBLIP
