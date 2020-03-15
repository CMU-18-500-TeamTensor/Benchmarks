from benchmark import bench_suite
from dimensions import conv_dimensions, pool_dimensions

import torch
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

import time

def now():
    return int(round(time.time() * 1000))

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    '''


    class Flatten(nn.Module):
        def forward(self, input):
            return input.view(-1, int(input.numel()/input.shape[0]))

    class G_M14(nn.Module):
        def __init__(self):
            super(G_M14, self).__init__()
            self.conv = nn.Conv2d(1, 30, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.flatten = Flatten()

            # Calculate the output size of the Flatten Layer
            # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
            c_out, h_out, w_out = conv_dimensions(1, 32, 32, 30, 1, 0, 5, 5)
            c_out, h_out, w_out = pool_dimensions(30, h_out, w_out, 2)
            flatten_size = c_out * h_out * w_out

            self.fc1 = nn.Linear(flatten_size, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv(x)))
            x = self.flatten(x)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

        def model_string(self):
            return "Grayscale pipeline model 14"
    '''


    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    start_time = now()
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            #inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    final_time = now()
    print("total time is: ", (final_time-start_time))
    print('Finished Training')
    return

#Need to do this for windows
if __name__ == '__main__':
    main()