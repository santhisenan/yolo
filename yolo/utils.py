import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def mpl_imshow_tensor(img, nrow=4):
    if type(img) == torch.Tensor and img.shape[0] > 1:
        img = make_grid(img, nrow)
    return plt.imshow(img.permute(1, 2, 0).numpy())
