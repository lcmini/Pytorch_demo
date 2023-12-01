import cv2
import torch
import torchvision.transforms as transforms
from torch import nn
import numpy as np
from torch.fx.experimental.migrate_gradual_types.constraint import F
from torch.nn import Sequential


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
# print(model)

# 转换模型为ONNX格式
dummy_input = torch.randn(32, 3, 32, 32)
dummy_input = dummy_input.to("cuda")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dummy_input.to(device)

input_names = ['input']
output_names = ['output']
torch.onnx.export(model, dummy_input, 'model.onnx', verbose=True, input_names=input_names, output_names=output_names)

# 加载ONNX模型
net = cv2.dnn.readNetFromONNX('model.onnx')

# 加载图像
image = cv2.imread('/home/lc/Torch_test/example/test.jpg')

# 图像预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = transform(image).unsqueeze(0)
image = np.array(image)

# 进行推理
# blob = cv2.dnn.blobFromTensor(image.numpy(), size=(224, 224))
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(32, 32), mean=[20, 30, 40])

net.setInput(blob)
output = net.forward()

# 处理输出结果
output = torch.from_numpy(output)
_, preds = torch.max(output, 1)

# 显示结果
print(preds.item())
# print(not preds.item)
