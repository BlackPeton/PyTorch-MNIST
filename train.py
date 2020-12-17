import math
import torch
import os

from utils import dataLoader
from cnn import simple_net


DATA_PATH = './DataSets'
MODEL_PATH = './Models'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
EPOCH = 40

print('Loading train set...')
train_set = dataLoader.loadTrain_set(DATA_PATH)
train_data = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
BATCH_NUM = math.ceil(len(train_set)/BATCH_SIZE)
print('Using ', DEVICE)

# 建立模型并载入设备
model = simple_net.MyCNN().to(DEVICE)
# 定义损失及优化器
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

print('\n-----------------\n'
      'Num of epoch: {}\n'
      'Batch size: {}\n'
      'Num of batch: {}'.format(
          EPOCH, BATCH_SIZE, BATCH_NUM))
print('-----------------\n')
print('Start training...')
# 训练
for epoch in range(EPOCH):
    print('Training epoch {}/{}'.format(epoch+1, EPOCH))
    num_correct = 0
    val_loss = 0
    for batch_idx, (images, labels) in enumerate(train_data):
        num_correct_batch = 0
        val_loss_batch = 0
        # 注意这里的images和labels均为一个batch的图片和标签
        images = images.to(DEVICE)  # BATCH_SIZE*28*28
        labels = labels.to(DEVICE)  # BATCH_SIZE*1

        outputs = model(images)
        pred = torch.max(outputs, 1)[1]  # 这一步将给出每张图片的分类结果，BATCH_SIZE*1
        optimizer.zero_grad()
        loss = cost(outputs, labels)
        loss.backward()
        optimizer.step()
        val_loss_batch += loss.data
        val_loss += val_loss_batch
        num_correct_batch += (pred == labels).sum().item()
        num_correct += num_correct_batch
        print('Batch {}/{}, Loss: {:.6f}, Accuracy: {:.6f}%'.format(batch_idx + 1,
                                                                    BATCH_NUM, val_loss_batch / BATCH_SIZE, 100 * num_correct_batch / BATCH_SIZE))
    print('Epoch {}: Loss: {:.6f}, Accuracy: {:.6f}%\n'.format(
        epoch + 1, val_loss / len(train_set), 100 * num_correct / len(train_set)))
# 保存整个网络
print('Saving the model...')
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
torch.save(model, MODEL_PATH + '/MyCNN_MNIST.pkl')
