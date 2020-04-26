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

# The true function we are modeling
def get_true_output(x):
    y = torch.zeros(10)

    y[0] = x[0]
    y[1] = x[0]**2 + x[1]
    y[2] = x[2]
    y[3] = x[2] * x[3]
    y[4] = x[3]**2 * x[4]
    y[5] = x[5]
    y[6] = y[6]**3
    y[7] = y[7]
    y[8] = torch.sqrt(y[8])
    y[9] = y[9] if y[9] > 0 else 0.0
    return y

# Full color Data Pipeline -- process data points as they are
def full_color(x):
    return x

# Grayscale Data Pipeline -- convert points to 1-channel images
def grayscale(x):
    return x[:10] - x[10:]

"""

    Full Color Models

"""

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(-1, int(input.numel()/input.shape[0]))


class FC_M1(nn.Module):
    def __init__(self):
        super(FC_M1, self).__init__()
        self.relu = nn.ReLU()

        # Calculate the output size of the Flatten Layer
        # conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width)
        self.fc = nn.Linear(20, 30)
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 1"

class FC_M2(nn.Module):
    def __init__(self):
        super(FC_M2, self).__init__()
        self.fc = nn.Linear(20, 40)
        self.fc2 = nn.Linear(40, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 2"

class FC_M3(nn.Module):
    def __init__(self):
        super(FC_M3, self).__init__()

        self.fc = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 3"

class FC_M4(nn.Module):
    def __init__(self):
        super(FC_M4, self).__init__()
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def model_string(self):
        return "Full Color pipeline model 4"

class FC_M5(nn.Module):
    def __init__(self):
        super(FC_M5, self).__init__()

        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 60)
        self.fc3 = nn.Linear(60, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 5"

class FC_M6(nn.Module):
    def __init__(self):
        super(FC_M6, self).__init__()
        self.fc1 = nn.Linear(20, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 6"

class FC_M7(nn.Module):
    def __init__(self):
        super(FC_M7, self).__init__()
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 7"

class FC_M8(nn.Module):
    def __init__(self):
        super(FC_M8, self).__init__()
        self.fc1 = nn.Linear(20, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 8"

class FC_M9(nn.Module):
    def __init__(self):
        super(FC_M9, self).__init__()
        self.fc1 = nn.Linear(20, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 9"

class FC_M10(nn.Module):
    def __init__(self):
        super(FC_M10, self).__init__()
        self.fc1 = nn.Linear(20, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 10"

class FC_M11(nn.Module):
    def __init__(self):
        super(FC_M11, self).__init__()
        self.fc = nn.Linear(20, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 11"

class FC_M12(nn.Module):
    def __init__(self):
        super(FC_M12, self).__init__()

        self.fc = nn.Linear(20, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 12"

class FC_M13(nn.Module):
    def __init__(self):
        super(FC_M13, self).__init__()

        self.fc = nn.Linear(20, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 13"


#NOTE: the relu layer has to be defined in the constructor otherwise
#tensorfpga will not be able to corretly count the number of layers
class FC_M14(nn.Module):
    def __init__(self):
        super(FC_M14, self).__init__()
        self.fc1 = nn.Linear(20, 500)
        self.fc2 = nn.Linear(500, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Full Color pipeline model 14"

class FC_M15(nn.Module):
    def __init__(self):
        super(FC_M15, self).__init__()

        self.fc1 = nn.Linear(20, 500)
        self.fc2 = nn.Linear(500, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
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
        self.fc = nn.Linear(10, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 1"

class G_M2(nn.Module):
    def __init__(self):
        super(G_M2, self).__init__()
        self.fc = nn.Linear(10, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 2"

class G_M3(nn.Module):
    def __init__(self):
        super(G_M3, self).__init__()

        self.fc = nn.Linear(10, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 3"

class G_M4(nn.Module):
    def __init__(self):
        super(G_M4, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 4"

class G_M5(nn.Module):
    def __init__(self):
        super(G_M5, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        #x = self.relu()
        #x = self.relu(self.conv2(x))
        #x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 5"

class G_M6(nn.Module):
    def __init__(self):
        super(G_M6, self).__init__()
        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 6"

class G_M7(nn.Module):
    def __init__(self):
        super(G_M7, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Grayscalw pipeline model 7"

class G_M8(nn.Module):
    def __init__(self):
        super(G_M8, self).__init__()
        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 8"

class G_M9(nn.Module):
    def __init__(self):
        super(G_M9, self).__init__()
        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 9"

class G_M10(nn.Module):
    def __init__(self):
        super(G_M10, self).__init__()

        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 10"

class G_M11(nn.Module):
    def __init__(self):
        super(G_M11, self).__init__()

        self.fc = nn.Linear(10, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 11"

class G_M12(nn.Module):
    def __init__(self):
        super(G_M12, self).__init__()
        self.fc = nn.Linear(10, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 12"

class G_M13(nn.Module):
    def __init__(self):
        super(G_M13, self).__init__()
        self.fc = nn.Linear(10, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 13"

class G_M14(nn.Module):
    def __init__(self):
        super(G_M14, self).__init__()
        self.fc1 = nn.Linear(10, 500)
        self.fc2 = nn.Linear(500, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 14"

class G_M15(nn.Module):
    def __init__(self):
        super(G_M15, self).__init__()
        self.fc1 = nn.Linear(10, 500)
        self.fc2 = nn.Linear(500, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def model_string(self):
        return "Grayscale pipeline model 15"

"""

    DP<->Model relationships
"""

#bench_suite = {"full_color": (full_color, [FC_M1, FC_M14])}
#bench_suite = {"grayscale": (grayscale, [G_M14])}




bench_suite = {"full_color": (full_color, [FC_M1, FC_M2, FC_M3, FC_M4, FC_M5, FC_M6, FC_M7, FC_M8, FC_M9, FC_M10, FC_M11, FC_M12, FC_M13, FC_M14, FC_M15]),
            "grayscale":  (grayscale,  [G_M1,  G_M2,  G_M3,  G_M4,  G_M5,  G_M6, G_M7,  G_M8,  G_M9,  G_M10,  G_M11,  G_M12,  G_M13,  G_M14,  G_M15])}



