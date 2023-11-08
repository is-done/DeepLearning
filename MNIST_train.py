import torch
import torch.nn as nn
from torchvision import datasets
from model import Mnist_model
import torch.optim as optim
import torchvision.transforms as transforms


epoch=20
batch_size=128
save_path = './pth/MNIST.pth'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])
train_dataset = datasets.MNIST('./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Mnist_model().to(device)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

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
