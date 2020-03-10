# Benchmarks
benchmark.py: File that holds 30 models (15 Full Color, 15 Gray scale), also delcares dictionary at the bottom which is used
to access all models stated in file

example_code.py: Example code for what a user would write when training a given set of a models on some datatset

dimensions.py: File used to calculate input/output size of convolutional and pool layers

tensorfpga.py: Rough skeleton of functions that the Python API will use to acquire boards, distribute and push models
to boards, and manage workflow

cpu_sequential.py: Sample code that is mostly pulled from the PyTorch tutorial page on how to train a given model
Currrently configured to train the 30 models specified in benchmark.py and to train on local machine's cpu

gpu_sequential.py: Sample code that is mostly pulled from the PyTorch tutorial page on how to train a given model
Currrently configured to train the 30 models specified in benchmark.py and to train on local machine's GPU


