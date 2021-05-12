import torch 
from torch.utils.data import DataLoader
import torchvision.datasets as dsets 
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable

batch_size = 100
# MNIST dataset
train_dataset = dsets.MNIST(root='./pymnist', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./pymnist', train=False, transform=transforms.ToTensor(), download=True)
# load_data
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# original_data
print("train_data:", train_dataset.train_data.size())
print("train_labels:", train_dataset.train_labels.size())
print("test_data:", test_dataset.test_data.size())
print("test_labels:", test_dataset.test_labels.size())
# shuffle batch_size data
print("batch_size:", train_loader.batch_size)
print("load_train_data:", train_loader.dataset.train_data.shape)
print("load_train_labels:", train_loader.dataset.train_labels.shape)

input_size = 784
hidden_size = 500
num_classes = 10

# #定义神经网络模型
class Neural_net(nn.Module):
    def __init__(self, input_num, hidden_size, output_num):
        super(Neural_net, self).__init__()
        self.layers1 = nn.Linear(input_num, hidden_size)
        self.layers2 = nn.Linear(hidden_size, output_num)

    def forward(self, x):
        out = self.layers1(x)
        out = torch.relu(out)
        out = self.layers2(out)
        return out
net = Neural_net(input_size, hidden_size, num_classes)
print(net)

learning_rate = 1e-1
num_epoches = 5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
for epoch in range(num_epoches):
    print("current epoch = {}".format(epoch))
    for i, (images,labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        outputs = net(images)
        loss = criterion(outputs, labels)  # calculate loss
        optimizer.zero_grad()  # clear net state before backward
        loss.backward()       
        optimizer.step()   # update parameters

        if i%100 == 0:
            print("current loss = %.5f" %loss.item())
print("finished training")

total = 0
correct = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    labels = Variable(labels)
    outputs = net(images)

    _,predicts = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicts == labels).sum()
print("Accuracy = %.2f" %(100*correct/total))

torch.save(net, "mnist_torch.pth")
net1 = torch.load("mnist_torch.pth")
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    labels = Variable(labels)
    outputs = net1(images)

    _,predicts = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicts == labels).sum()
print("Accuracy = %.2f" %(100*correct/total))

