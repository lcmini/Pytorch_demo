import os
import torch
from PIL import Image
from torch import nn
from torch.nn import Conv2d, Sequential, MaxPool2d, Flatten, ReLU
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


class Mydata(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, item):
        toTensor = transforms.ToTensor()
        trans_resize = transforms.Resize((32, 32))
        img_name = self.img_path[item]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path).convert('RGB')
        img = trans_resize(img)
        img = toTensor(img)
        label = self.label_dir
        if label == "ants":
            new_label = 0
        else:
            new_label = 1
        return img, new_label

    def __len__(self):
        return len(self.img_path)


# 构建神经网络
class Net(nn.Module):  # 定义网络模块
    def __init__(self):
        super(Net, self).__init__()
        self.model1 = Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 2)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

root_dir = "mydata/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = Mydata(root_dir, ants_label_dir)
bees_dataset = Mydata(root_dir, bees_label_dir)
train_dataset = ants_dataset + bees_dataset
test_data = bees_dataset

train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
test_dataloader = DataLoader(dataset=bees_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
test_data_size = len(test_data)

writer = SummaryWriter("logs")

animal = Net()
animal = animal.to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

learning_rate = 5e-3
optimizer = torch.optim.SGD(animal.parameters(), lr=learning_rate)

total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 1000

for i in range(epoch):
    print("The " + str(i + 1) + " turn train start")
    # train start
    animal.train()
    for data in train_dataloader:
        img, targets = data
        img = img.to(device)
        targets = targets.to(device)

        outputs = animal(img)
        # print(outputs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        writer.add_scalar("train_loss", loss.item(), total_train_step)

    # test start
    animal.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            img, labels = data
            targets = labels
            img = img.to(device)
            targets = targets.to(device)
            outputs = animal(img)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体集上的Loss:{}".format(total_test_loss))
    print("整体数据集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(animal, "./weights/torch.pth")

writer.close()
