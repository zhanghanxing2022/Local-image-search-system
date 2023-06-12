import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import os
# 数据集路径
image_features_dir = 'data/image_features'
descriptions_features_dir = 'data/descriptions_features'

import random
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_features_dir, descriptions_features_dir,k):
        self.image_features_dir = image_features_dir
        self.descriptions_features_dir = descriptions_features_dir
        self.k = k
        # 获取文件列表


        a_folder = self.image_features_dir
        b_folder = self.descriptions_features_dir

        # 获取a文件夹中的所有文件名
        a_files = os.listdir(a_folder)

        # 获取b文件夹中的所有文件名
        b_files = os.listdir(b_folder)

        # 初始化结果列表
        self.image_files  = []
        self.description_files  =[]
        # 遍历a文件夹中的文件
        for a_file in a_files:
            # 提取文件名的前缀（去除".pt"后缀）
            prefix = os.path.splitext(a_file)[0]

            # 检查是否有5个对应的文件
            has_5_files = True
            five_files = []
            for i in range(5):
                x_file = f"{prefix}_{i}.pt"
                five_files.append(x_file)
                if x_file not in b_files:
                    has_5_files = False
                    break

            # 如果有5个对应的文件，则将x文件名添加到结果列表中
            if has_5_files:
                self.image_files.append(prefix + ".pt")
                
                self.description_files.append(five_files)

        # 打印结果列表
        print(self.image_files)
        print(len(self.image_files))
        print(self.description_files)
        print(len(self.description_files))
        self.non_matching = []
        for i  in range(len(self.image_files)):
            non_matching_indices = random.sample(range(len(self.description_files)), self.k)
            temp=[]
            for indices in non_matching_indices:
                temp.append(self.description_files[indices][1])
            self.non_matching.append(temp)

        


    def __getitem__(self, index):
        # 加载图片特征
        
        image_index = int(index/(5+self.k))
        image_feature_path = os.path.join(self.image_features_dir, self.image_files[image_index])
        image_feature = torch.load(image_feature_path)
        
        description_index = index%(self.k+5)
        description_feature_path =""
        if description_index < 5:
            description_feature_path = os.path.join(self.descriptions_features_dir, self.description_files[image_index][description_index])
            target = 1.0
        else:
            description_feature_path = os.path.join(self.descriptions_features_dir, self.non_matching[image_index][description_index-5])
            target = 0.0

        # 加载对应的文字特征向量
        
        description_feature = torch.load(description_feature_path)
        
        return image_feature.squeeze(0), description_feature.squeeze(0), torch.Tensor([target])
    
    def __len__(self):
        return len(self.image_files)*(5+self.k)

k = 10
# 创建数据集实例
dataset = CustomDataset(image_features_dir, descriptions_features_dir,k)

# 划分训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
a,b,c = train_dataset[0]
print(a.shape)
print(b.shape)
print(c)


# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义网络模型
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # 图片特征处理
        self.cnn = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3)
        )
        
        # 描述特征处理
        self.fc_description = nn.Sequential(
            nn.Linear(30 * 512, 256),
            nn.ReLU()
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(512 , 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image_feature, description_features):
        # 图片特征处理
        image_feature = self.cnn(image_feature)
        image_feature = image_feature.view(image_feature.size(0), -1)
        
        # 描述特征处理
        description_features = description_features.view(description_features.size(0), -1)
        description_features = self.fc_description(description_features)
        
        # 连接图片特征和描述特征
        '''
        torch.Size([32, 256])
        torch.Size([32, 256])
        '''
        combined_features = torch.cat([image_feature, description_features], dim=1)
        
        # 预测
        output = self.fc(combined_features)
        
        return output

import torch.optim as optim

# 定义训练函数
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        running_loss = 0.0
        for image_feature, description_features, targets in train_loader:
            # 将输入数据移动到设备上（例如GPU）
            # image_feature = image_feature.to(torch.device)
            # description_features = description_features.to(torch.device)
            # targets = targets.to(torch.device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(image_feature, description_features)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计损失
            running_loss += loss.item() * image_feature.size(0)
        
        # 打印每个epoch的损失
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 创建模型实例
model = Net()

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置设备（例如GPU）并移动模型到设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练模型
num_epochs = 10
train(model, train_loader, criterion, optimizer, num_epochs)

# 选择要保存的模型状态
model_state = model.state_dict()

# 保存模型状态到文件
save_path = '/save/model.pth'
torch.save(model_state, save_path)