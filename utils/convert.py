import matplotlib.pyplot as pylt
import torch

from . import dataLoader

def toImages(dataset_path, images_path, num_train, num_test):
    train_set = dataLoader.loadTrain_set(dataset_path)
    test_set = dataLoader.loadTest_set(dataset_path)
    train_data = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=60000, shuffle=False)
    test_data = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=10000, shuffle=False)
    images_train, labels_train = train_data
    images_test, labels_test = test_data
    for i in range(num_train):
        pylt.imsave(images_path + '/train/' + i + '.png', images_train[i], cmap='Gray')
    for j in range(num_test):
        pylt.imsave(images_path + '/test/' + j + '.png', images_test[j], cmap='Gray')