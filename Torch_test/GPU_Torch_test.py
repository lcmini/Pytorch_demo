import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Sequential

image_path = "/home/lc/Torch_test/example/test.jpg"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device1 = torch.device('cpu')
# 加载模型
model = Net()
model = torch.load('/home/lc/Torch_test/weights/torch.pth', map_location=torch.device("cuda"))
model.eval()
print(model)

# image = torch.reshape(image, (-1,1,32,32))
image = torch.reshape(image, (-1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image.to("cuda"))
print(output)

print(output.argmax(1))
