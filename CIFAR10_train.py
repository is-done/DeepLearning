import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from model import CIFAR10_model
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

epoch = 20
batch_size = 128
save_path = './pth/CIFAR10.pth'

transform = transforms.Compose(
    [torchvision.transforms.Pad(4),
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.RandomCrop(32),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
train_dataset = datasets.CIFAR10('./data/CIFAR10', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = CIFAR10_model().to(device)
loss_function = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=epoch,
                                                steps_per_epoch=len(train_loader))

for i in range(epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    for (step, data) in enumerate(train_loader, start=0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        scheduler.step()
    print(f"epoch:{i + 1},running_loss:{running_loss / batch_size}")
torch.save(net.state_dict(), save_path)
print('Finished Training')

net.load_state_dict(torch.load(save_path))
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images: {accuracy}%')
