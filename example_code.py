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

train_models(dpm, data)

# Retrieve accuracy from each metric
for model in dpm.pipelines[dp1_id].models:
    model.get_metric("accuracy")
