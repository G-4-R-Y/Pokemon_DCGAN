import torch
from torch import nn
import torchvision
import warnings

from config import Config

# actually resizes to 32x32, in hope we get better results as the problem gets simpler
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5, 0.5)
])

pokemon.transforms = transforms

data_iter = torch.utils.data.DataLoader(pokemon, batch_size=Config.batch_size, shuffle=True, num_workers=d2l.get_dataloader_workers())