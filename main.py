from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import numpy as np
import torchvision.transforms as T

if __name__ == '__main__':
    plt.rcParams["savefig.bbox"] = 'tight'
    orig_img = Image.open(Path(r'C:\Users\Sahin\PycharmProjects\pytorch-CycleGAN-and-pix2pix\datasets\Noisy2Clean\trainA\clean.png'))
    torch.manual_seed(0)
    random_cropped_128x128 = [T.RandomCrop(size=128)(orig_img) for i in range(16)]
    #random_cropped_64x64
    #random_cropped_32x32
    for img in random_cropped_128x128:
        plt.imshow(img)
