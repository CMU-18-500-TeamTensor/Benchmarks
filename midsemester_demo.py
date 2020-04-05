'''
This file contains code that will be used for the mid semester demo.
The code below will be used to show that the three subsystems have been connected
together. In this scenario we will only be using one model, and will only be passing the 
model to one connected board. The model that we use in this scenario is FC_M14.


'''
# Anything from the tensorfpga library is written by us, and not by the
# end user.
from tensorfpga import DataPipelineManager
from tensorfpga import train_models


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
	dpm = DataPipelineManager(1)

	# add_pipeline() returns the int ID of the data pipeline, as seen by the
	# FPGA
	# buffer size is the number of samples can be stored in the Data Source Server
	# buffer.
	#dp1_id = dpm.add_pipeline(dp1, name="grayscale", buffer_size=10)
	#dp2_id = dpm.add_pipeline(dp2, name="full_color", buffer_size=10)

	#user function to pull dataset
	data = get_CIFAR10_dataset() # iterable over tuples (x, y)

	#grab the whole model list, in this scenario just one model
	model_list = bench_suite

	pipeline_fn, model1 = model_list["full_color"]
	dp_id = dpm.add_pipeline(pipeline_fn, "full_color", 10)
	
	print("What is model1: ", model1)
	#User specificed which model goes to which pipeline
	#In this scenario, we are adding Full Color Model 14 to the full color pipeline
	dpm.add_model(model1, dp_id)
	
	#Once all the models have been added to their respective pipelines and the data
	#has been pulled, we begin training the models
	train_models(dpm, data)

	# Retrieve accuracy from each metric
	#for model in dpm.pipelines[dp1_id].models:
	#    model.get_metric("accuracy")

if __name__ == '__main__':
	main()