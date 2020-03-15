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
    return transforms.ToTensor()(grayscale_transform(transforms.ToPILImage(mode='RGB')(x)))

"""

    Full Color Models

"""

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(-1, int(input.numel()/input.shape[0]))


class FC_M1(nn.Module):
    def __init__(self):
        super(FC_M1, self).__init__()
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
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 1"

class FC_M2(nn.Module):
    def __init__(self):
        super(FC_M2, self).__init__()
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
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 2"

class FC_M3(nn.Module):
    def __init__(self):
        super(FC_M3, self).__init__()
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
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 3"

class FC_M4(nn.Module):
    def __init__(self):
        super(FC_M4, self).__init__()
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
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 4"

class FC_M5(nn.Module):
    def __init__(self):
        super(FC_M5, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.conv2 = nn.Conv2d(5, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(3, 32, 32, 5, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        c_out, h_out, w_out = conv_dimensions(5, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(6, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc1 = nn.Linear(flatten_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 5"

class FC_M6(nn.Module):
    def __init__(self):
        super(FC_M6, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.conv2 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(3, 32, 32, 3, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(3, h_out, w_out, 2)
        c_out, h_out, w_out = conv_dimensions(3, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(6, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc1 = nn.Linear(flatten_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 6"

class FC_M7(nn.Module):
    def __init__(self):
        super(FC_M7, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.conv2 = nn.Conv2d(3, 5, 5)
        self.conv3 = nn.Conv2d(5, 6, 5)
        self.conv4 = nn.Conv2d(6, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(3, 32, 32, 3, 1, 0, 5, 5)
        c_out, h_out, w_out = conv_dimensions(3, h_out, w_out, 5, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        c_out, h_out, w_out = conv_dimensions(5, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = conv_dimensions(6, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(6, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc1 = nn.Linear(flatten_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 7"

class FC_M8(nn.Module):
    def __init__(self):
        super(FC_M8, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.conv2 = nn.Conv2d(3, 5, 5)
        self.conv3 = nn.Conv2d(5, 6, 5)
        self.conv4 = nn.Conv2d(6, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(3, 32, 32, 3, 1, 0, 5, 5)
        c_out, h_out, w_out = conv_dimensions(3, h_out, w_out, 5, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        c_out, h_out, w_out = conv_dimensions(5, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = conv_dimensions(6, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(6, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc1 = nn.Linear(flatten_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 8"

class FC_M9(nn.Module):
    def __init__(self):
        super(FC_M9, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.conv2 = nn.Conv2d(3, 5, 5)
        self.conv3 = nn.Conv2d(5, 6, 5)
        self.conv4 = nn.Conv2d(6, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(3, 32, 32, 3, 1, 0, 5, 5)
        c_out, h_out, w_out = conv_dimensions(3, h_out, w_out, 5, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        c_out, h_out, w_out = conv_dimensions(5, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = conv_dimensions(6, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(6, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc1 = nn.Linear(flatten_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 9"

class FC_M10(nn.Module):
    def __init__(self):
        super(FC_M10, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.conv2 = nn.Conv2d(3, 5, 5)
        self.conv3 = nn.Conv2d(5, 6, 5)
        self.conv4 = nn.Conv2d(6, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(3, 32, 32, 3, 1, 0, 5, 5)
        c_out, h_out, w_out = conv_dimensions(3, h_out, w_out, 5, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        c_out, h_out, w_out = conv_dimensions(5, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = conv_dimensions(6, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(6, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc1 = nn.Linear(flatten_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 10"

class FC_M11(nn.Module):
    def __init__(self):
        super(FC_M11, self).__init__()
        self.conv = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(3, 32, 32, 10, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(10, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc = nn.Linear(flatten_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 11"

class FC_M12(nn.Module):
    def __init__(self):
        super(FC_M12, self).__init__()
        self.conv = nn.Conv2d(3, 25, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(3, 32, 32, 25, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(25, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc = nn.Linear(flatten_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 12"

class FC_M13(nn.Module):
    def __init__(self):
        super(FC_M13, self).__init__()
        self.conv = nn.Conv2d(3, 30, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(3, 32, 32, 30, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(30, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc = nn.Linear(flatten_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 13"

class FC_M14(nn.Module):
    def __init__(self):
        super(FC_M14, self).__init__()
        self.conv = nn.Conv2d(3, 30, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(3, 32, 32, 30, 1, 0, 5, 5)
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
        return "Full Color pipeline model 14"

class FC_M15(nn.Module):
    def __init__(self):
        super(FC_M15, self).__init__()
        self.conv1 = nn.Conv2d(3, 30, 5)
        self.conv2 = nn.Conv2d(30, 60, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(3, 32, 32, 30, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(30, h_out, w_out, 2)
        c_out, h_out, w_out = conv_dimensions(30, h_out, w_out, 60, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(60, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc1 = nn.Linear(flatten_size, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 15"

"""

    Grayscale Models

"""

class G_M1(nn.Module):
    def __init__(self):
        super(G_M1, self).__init__()
        self.conv = nn.Conv2d(1, 1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(1, 32, 32, 1, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(1, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc = nn.Linear(flatten_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 1"

class G_M2(nn.Module):
    def __init__(self):
        super(G_M2, self).__init__()
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
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 2"

class G_M3(nn.Module):
    def __init__(self):
        super(G_M3, self).__init__()
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
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 3"

class G_M4(nn.Module):
    def __init__(self):
        super(G_M4, self).__init__()
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
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 4"

class G_M5(nn.Module):
    def __init__(self):
        super(G_M5, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5)
        self.conv2 = nn.Conv2d(5, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(1, 32, 32, 5, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        c_out, h_out, w_out = conv_dimensions(5, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(6, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc1 = nn.Linear(flatten_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 5"

class G_M6(nn.Module):
    def __init__(self):
        super(G_M6, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.conv2 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(1, 32, 32, 3, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(3, h_out, w_out, 2)
        c_out, h_out, w_out = conv_dimensions(3, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(6, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc1 = nn.Linear(flatten_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 6"

class G_M7(nn.Module):
    def __init__(self):
        super(G_M7, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.conv2 = nn.Conv2d(3, 5, 5)
        self.conv3 = nn.Conv2d(5, 6, 5)
        self.conv4 = nn.Conv2d(6, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(1, 32, 32, 3, 1, 0, 5, 5)
        c_out, h_out, w_out = conv_dimensions(3, h_out, w_out, 5, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        c_out, h_out, w_out = conv_dimensions(5, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = conv_dimensions(6, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(6, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc1 = nn.Linear(flatten_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Grayscalw pipeline model 7"

class G_M8(nn.Module):
    def __init__(self):
        super(G_M8, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.conv2 = nn.Conv2d(3, 5, 5)
        self.conv3 = nn.Conv2d(5, 6, 5)
        self.conv4 = nn.Conv2d(6, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(1, 32, 32, 3, 1, 0, 5, 5)
        c_out, h_out, w_out = conv_dimensions(3, h_out, w_out, 5, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        c_out, h_out, w_out = conv_dimensions(5, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = conv_dimensions(6, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(6, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc1 = nn.Linear(flatten_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 8"

class G_M9(nn.Module):
    def __init__(self):
        super(G_M9, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.conv2 = nn.Conv2d(3, 5, 5)
        self.conv3 = nn.Conv2d(5, 6, 5)
        self.conv4 = nn.Conv2d(6, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(1, 32, 32, 3, 1, 0, 5, 5)
        c_out, h_out, w_out = conv_dimensions(3, h_out, w_out, 5, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        c_out, h_out, w_out = conv_dimensions(5, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = conv_dimensions(6, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(6, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc1 = nn.Linear(flatten_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 9"

class G_M10(nn.Module):
    def __init__(self):
        super(G_M10, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.conv2 = nn.Conv2d(3, 5, 5)
        self.conv3 = nn.Conv2d(5, 6, 5)
        self.conv4 = nn.Conv2d(6, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(1, 32, 32, 3, 1, 0, 5, 5)
        c_out, h_out, w_out = conv_dimensions(3, h_out, w_out, 5, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(5, h_out, w_out, 2)
        c_out, h_out, w_out = conv_dimensions(5, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = conv_dimensions(6, h_out, w_out, 6, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(6, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc1 = nn.Linear(flatten_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 10"

class G_M11(nn.Module):
    def __init__(self):
        super(G_M11, self).__init__()
        self.conv = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(1, 32, 32, 10, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(10, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc = nn.Linear(flatten_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 11"

class G_M12(nn.Module):
    def __init__(self):
        super(G_M12, self).__init__()
        self.conv = nn.Conv2d(1, 25, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(1, 32, 32, 25, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(25, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc = nn.Linear(flatten_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 12"

class G_M13(nn.Module):
    def __init__(self):
        super(G_M13, self).__init__()
        self.conv = nn.Conv2d(1, 30, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        c_out, h_out, w_out = conv_dimensions(1, 32, 32, 30, 1, 0, 5, 5)
        c_out, h_out, w_out = pool_dimensions(30, h_out, w_out, 2)
        flatten_size = c_out * h_out * w_out

        self.fc = nn.Linear(flatten_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 13"

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

class G_M15(nn.Module):
    def __init__(self):
        super(G_M15, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, 5)
        self.conv2 = nn.Conv2d(30, 60, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()

        # Calculating total memory usage:
        print("Model weights (units are weights, where one weight is 4 bytes):")
        print("Note: These numbers are already doubled to account for gradient space")
        print(1*30*5*5*2)
        print(30*60*5*5*2)


        print("\nIntermediate Layers:")
        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        print(1*32*32)
        c_out, h_out, w_out = conv_dimensions(1, 32, 32, 30, 1, 0, 5, 5)
        print(c_out*h_out*w_out)
        c_out, h_out, w_out = pool_dimensions(30, h_out, w_out, 2)
        print(c_out*h_out*w_out)
        c_out, h_out, w_out = conv_dimensions(30, h_out, w_out, 60, 1, 0, 5, 5)
        print(c_out*h_out*w_out)
        c_out, h_out, w_out = pool_dimensions(60, h_out, w_out, 2)
        print(c_out*h_out*w_out)
        flatten_size = c_out * h_out * w_out
        print(500)

        print("\nMore model weights:")
        self.fc1 = nn.Linear(flatten_size, 500)
        print(flatten_size * 500 * 2)
        self.fc2 = nn.Linear(500, 10)
        print(500*10*2)



    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 15"

"""

    DP<->Model relationships

"""
#bench_suite = {"full_color": (full_color, [FC_M15])}


bench_suite = {"full_color": (full_color, [FC_M1, FC_M2, FC_M3, FC_M4, FC_M5, FC_M6, FC_M7, FC_M8, FC_M9, FC_M10, FC_M11, FC_M12, FC_M13, FC_M14, FC_M15]),
               "grayscale":  (grayscale,  [G_M1,  G_M2,  G_M3,  G_M4,  G_M5,  G_M6, G_M7,  G_M8,  G_M9,  G_M10,  G_M11,  G_M12,  G_M13,  G_M14,  G_M15])}
               
