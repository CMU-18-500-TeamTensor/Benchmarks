# This file contains example code that a user would write to train a set of
# models on a given data set and retrieve metrics.

# Anything from the tensorfpga library is written by us, and not by the
# end user.
from tensorfpga import DataPipelineManager

# Anything from userprovided library is written by the end user.
#from userprovided import get_CIFAR10_dataset
from benchmark import bench_suite

# Third party library functio
from modelsummaryimport import summary

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

# User defined function to get the CIFAR10 dataset
def get_CIFAR10_dataset():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
    return trainloader

# Data pipelines defined by the user, written here for clarity
def dp1(x):
    return x.grayscale()

def dp2(x):
    return x

#going to be testing this on full color model 14
def main():
	dpm = DataPipelineManager(4)

	# add_pipeline() returns the int ID of the data pipeline, as seen by the
	# FPGA
	# buffer size is the number of samples can be stored in the Data Source Server
	# buffer.
	#dp1_id = dpm.add_pipeline(dp1, name="grayscale", buffer_size=10)
	#dp2_id = dpm.add_pipeline(dp2, name="full_color", buffer_size=10)
	dp_id = dpm.add_pipeline(dp2, "full_color", 10)

	#user function to pull dataset
	data = get_CIFAR10_dataset() # iterable over tuples (x, y)

	model_list = bench_suite
	print("What is model_list", model_list)
	#dpm.pipelines[dp1_id].add_model(model_list[0])
	

	#train_models(dpm, data)

	# Retrieve accuracy from each metric
	#for model in dpm.pipelines[dp1_id].models:
	#    model.get_metric("accuracy")

if __name__ == '__main__':
	main()