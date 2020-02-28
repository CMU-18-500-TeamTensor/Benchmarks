# This file contains example code that a user would write to train a set of
# models on a given data set and retrieve metrics.

# Anything from the tensorfpga library is written by us, and not by the
# end user.
from tensorfpga import DataPipelineManager

# Anything from userprovided library is written by the end user.
from userprovided import get_CIFAR10_dataset, Net1, Net2, Net3, Net4


dpm = DataPipelineManager()

# Data pipelines defined by the user, written here for clarity
def dp1(x):
    return x.grayscale()

def dp2(x):
    return x

# add_pipeline() returns the int ID of the data pipeline, as seen by the
# FPGA
# buffer size is the number of samples can be stored in the Data Source Server
# buffer.
dp1_id = dpm.add_pipeline(dp1, name="grayscale", buffer_size=10)
dp2_id = dpm.add_pipeline(dp2, name="full_color", buffer_size=10)

#user function to pull dataset
data = get_CIFAR10_dataset() # iterable over tuples (x, y)

models = [Net1(), Net2(), Net3(), Net4()]
dpm.pipelines[dp1_id].add_model(models[0])
dpm.pipelines[dp1_id].add_model(models[1])
dpm.pipelines[dp2_id].add_model(models[2])
dpm.pipelines[dp2_id].add_model(models[3])

for (x, y) in data:
    # if the data pipeline is full, then train_sample will stall until the data pipeline
    # has enough space for another sample
    dpm.pipelines[dp1_id].train_sample(x=x, y=y)
    dpm.pipelines[dp2_id].train_sample(x=x, y=y)

# Wait until training has completed
# .empty() returns true if there are no samples being processed within this data
# pipeline, including on the FPGA boards.
while !dpm.pipelines[dp1_id].empty() and !dpm.pipelines[dp2_id].empty():
    sleep(5)

# Retrieve accuracy from each metric
for model in dpm.pipelines[dp1_id].models:
    model.get_metric("accuracy")
