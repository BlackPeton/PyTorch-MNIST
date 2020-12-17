import matplotlib.pyplot as pylt
import torch
import os


def toImages(file, save_path, num):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data = torch.load(file)
    for i in range(num):
        img_name = save_path + '/' + str(i + 1) + '.png'
        pylt.imsave(img_name, data[0][i], cmap='Greys')
