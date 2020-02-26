import torch
import torchvision
import torchvision.transforms as transforms

from dimensions import conv_dimensions, pool_dimensions

import torch.nn as nn
import torch.nn.functional as F

"""

    DATA PIPELINE DEFINITIONS

    We're training on two data pipelines:
        - full color
        - grayscale

"""

# Full color Data Pipeline -- process data points as they are
def full_color(x):
    return x

# Grayscale Data Pipeline -- convert points to 1-channel images
def grayscale(x):
    grayscale_transform = transforms.Grayscale(num_output_channels=1)
    return grayscale_transform(x)

"""

    Full Color Models

"""

def Flatten(x):
    return x.view(-1, 16 * 5 * 5)


class FC_M1(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(3, 32, 32, 1, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(1, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc = nn.Linear(flatten_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.flatten()
        x = self.fc(x)
        return x

class FC_M2(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 5, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(3, 32, 32, 5, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc = nn.Linear(flatten_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.flatten()
        x = self.fc(x)
        return x

class FC_M3(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(3, 32, 32, 3, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(3, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc = nn.Linear(flatten_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.flatten()
        x = self.fc(x)
        return x

class FC_M4(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 5, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(3, 32, 32, 5, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc1 = nn.Linear(flatten_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.flatten()
        x = self.fc2(self.fc1(x))
        return x

class FC_M5(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.conv2 = nn.Conv2d(5, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(3, 32, 32, 5, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        c_out, h_out, w_out = conv_dimensions(5, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc1 = nn.Linear(flatten_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten()
        x = self.fc2(self.fc1(x))
        return x


"""

    Grayscale Models

"""

class G_M1(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(3, 32, 32, 1, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(1, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc = nn.Linear(flatten_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.flatten()
        x = self.fc(x)
        return x

class G_M2(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 5, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(1, 32, 32, 5, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc = nn.Linear(flatten_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.flatten()
        x = self.fc(x)
        return x

class G_M3(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(1, 32, 32, 3, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(3, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc = nn.Linear(flatten_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.flatten()
        x = self.fc(x)
        return x

class G_M4(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 5, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(1, 32, 32, 5, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc1 = nn.Linear(flatten_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.flatten()
        x = self.fc2(self.fc1(x))
        return x

class G_M5(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5)
        self.conv2 = nn.Conv2d(5, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(1, 32, 32, 5, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        c_out, h_out, w_out = conv_dimensions(5, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc1 = nn.Linear(flatten_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten()
        x = self.fc2(self.fc1(x))
        return x



"""

    DP<->Model relationships

"""

bench_suite = {"full_color": (full_color, [FC_M1, FC_M2, FC_M3, FC_M4, FC_M5]),
               "grayscale":  (grayscale,  [G_M1,  G_M2,  G_M3,  G_M4,  G_M5])}
