import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CIFAR10_model,Mnist_model


def main():
    transform = transforms.Compose(
        [transforms.Resize((28, 28)),
         transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)])

    classes = ('0', '1', '2', '3',
               '4', '5', '6', '7', '8', '9')

    net = Mnist_model()
    net.load_state_dict(torch.load('./pth/MNIST.pth'))

    im = Image.open('test/7.jpg')
    im = im.convert('L')
    im = im.split()[0]
    im = transform(im)
    im = torch.unsqueeze(im, dim=0)

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
