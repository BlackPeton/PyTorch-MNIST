import torch

from utils import dataLoader

DATA_PATH = './DataSets'
MODEL_PATH = './Models'
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Loading test set...')
test_set = dataLoader.loadTest_set(DATA_PATH)
test_data = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)
print('Using ', DEVICE)
print('Loading saved model...')
model = torch.load(MODEL_PATH + '/MyCNN_MNIST.pkl').to(DEVICE)
print('Testing...')

num_correct = 0
for images, labels in test_data:
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    outputs = model(images)
    pred = torch.max(outputs, 1)[1]
    num_correct += (pred == labels).sum().item()
print('Accuracy: {:.6f}%'.format(100 * num_correct / len(test_set)))
