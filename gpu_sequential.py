#Tests on local machine GPU (CURRENTLY NOT WORKING)


from benchmark import bench_suite

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

import time

def now():
    return int(round(time.time() * 1000))


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


#import torch.nn.functional as F


def main(device):
    num_epochs = 1

    birthday = now()
    num_models_trained = 0
    avg_time = 0
    trainloader = get_CIFAR10_dataset()

    statistics = []
    for data_pipeline in bench_suite.keys():
        print("Iterating through pipeline")
        pipeline_fn, models = bench_suite[data_pipeline]
        for model in models:
            print("Starting with first model")
            time_to_train = now()
            num_models_trained += 1
            # Train this model

            #net = model()
            net = model().to(device)
            

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            for epoch in range(num_epochs):  # loop over the dataset multiple times

                running_loss = 0.0
                print("Beginning training now")
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    #inputs,labels = data
                    #inputs = torch.stack([pipeline_fn(inputs[i]) for i in range(inputs.shape[0])])
                    
                    inputs, labels = data[0], data[1].to(device) #.to(device), data[1].to(device)
                    inputs = torch.stack([pipeline_fn(inputs[i]) for i in range(inputs.shape[0])]).to(device)

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
                time_to_finish = now() - time_to_train
                print('Finished Training: %s' % net.model_string())
                print('Time for model: ', time_to_finish);
                avg_time += time_to_finish
                statistics.append("Time taken: %.3f seconds, Loss value: %.3f" % (time_to_finish/1000, running_loss));

    time_taken = now() - birthday

    throughput = num_models_trained / time_taken
    avg_time = avg_time/(1000 * num_models_trained)
    print("Trained models with an average throughput of %f" % (throughput * 1000000))
    print("Trained models at an average time of %.3f seconds" % (avg_time))

    for i in range(len(statistics)):
        print("Model %d stats: %s" % (i+1,statistics[i]))

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    main(device)


'''
Statistics 
Hardware: NVIDIA GeForce RTX 2060 SUPER
Seems to be doing better on wider models
Trained models with an average throughput of 24.454581
Trained models at an average time of 40.866 seconds
Model 1 stats: Time taken: 28.239 seconds, Loss value: 966.509
Model 2 stats: Time taken: 24.933 seconds, Loss value: 732.629
Model 3 stats: Time taken: 25.145 seconds, Loss value: 804.955
Model 4 stats: Time taken: 30.565 seconds, Loss value: 727.075
Model 5 stats: Time taken: 38.075 seconds, Loss value: 749.385
Model 6 stats: Time taken: 48.607 seconds, Loss value: 852.541
Model 7 stats: Time taken: 50.306 seconds, Loss value: 852.214
Model 8 stats: Time taken: 55.584 seconds, Loss value: 902.778
Model 9 stats: Time taken: 60.663 seconds, Loss value: 1151.682
Model 10 stats: Time taken: 65.919 seconds, Loss value: 1150.963
Model 11 stats: Time taken: 24.625 seconds, Loss value: 674.633
Model 12 stats: Time taken: 24.847 seconds, Loss value: 617.159
Model 13 stats: Time taken: 25.726 seconds, Loss value: 606.477
Model 14 stats: Time taken: 32.640 seconds, Loss value: 566.784
Model 15 stats: Time taken: 37.984 seconds, Loss value: 590.729
Model 16 stats: Time taken: 29.991 seconds, Loss value: 1055.854
Model 17 stats: Time taken: 30.329 seconds, Loss value: 965.328
Model 18 stats: Time taken: 30.178 seconds, Loss value: 992.295
Model 19 stats: Time taken: 35.499 seconds, Loss value: 956.919
Model 20 stats: Time taken: 43.565 seconds, Loss value: 1139.789
Model 21 stats: Time taken: 54.319 seconds, Loss value: 1137.847
Model 22 stats: Time taken: 55.855 seconds, Loss value: 1151.514
Model 23 stats: Time taken: 61.021 seconds, Loss value: 1151.118
Model 24 stats: Time taken: 66.462 seconds, Loss value: 1151.826
Model 25 stats: Time taken: 71.953 seconds, Loss value: 1151.610
Model 26 stats: Time taken: 30.304 seconds, Loss value: 936.803
Model 27 stats: Time taken: 30.150 seconds, Loss value: 906.943
Model 28 stats: Time taken: 30.375 seconds, Loss value: 904.646
Model 29 stats: Time taken: 38.212 seconds, Loss value: 895.794
Model 30 stats: Time taken: 43.902 seconds, Loss value: 949.094




Hardware: Intel(R) Core(TM) i7-9700F CPU @ 3.00 GHz 3.00 GHz
Seems to be doing better on deeper models
Trained models with an average throughput of 24.921083
Trained models at an average time of 40.100 seconds
Model 1 stats: Time taken: 12.247 seconds, Loss value: 994.131
Model 2 stats: Time taken: 14.784 seconds, Loss value: 759.224
Model 3 stats: Time taken: 14.622 seconds, Loss value: 811.628
Model 4 stats: Time taken: 20.136 seconds, Loss value: 715.087
Model 5 stats: Time taken: 22.665 seconds, Loss value: 768.794
Model 6 stats: Time taken: 24.846 seconds, Loss value: 838.152
Model 7 stats: Time taken: 33.105 seconds, Loss value: 889.310
Model 8 stats: Time taken: 36.692 seconds, Loss value: 973.237
Model 9 stats: Time taken: 39.287 seconds, Loss value: 965.914
Model 10 stats: Time taken: 39.266 seconds, Loss value: 1150.552
Model 11 stats: Time taken: 19.270 seconds, Loss value: 697.697
Model 12 stats: Time taken: 24.998 seconds, Loss value: 623.859
Model 13 stats: Time taken: 27.572 seconds, Loss value: 618.720
Model 14 stats: Time taken: 151.185 seconds, Loss value: 606.125
Model 15 stats: Time taken: 83.843 seconds, Loss value: 576.875
Model 16 stats: Time taken: 18.548 seconds, Loss value: 1068.568
Model 17 stats: Time taken: 22.174 seconds, Loss value: 963.580
Model 18 stats: Time taken: 20.550 seconds, Loss value: 971.986
Model 19 stats: Time taken: 26.596 seconds, Loss value: 1005.134
Model 20 stats: Time taken: 28.523 seconds, Loss value: 1049.507
Model 21 stats: Time taken: 31.286 seconds, Loss value: 1139.938
Model 22 stats: Time taken: 36.644 seconds, Loss value: 1152.150
Model 23 stats: Time taken: 37.675 seconds, Loss value: 1150.964
Model 24 stats: Time taken: 38.490 seconds, Loss value: 1152.036
Model 25 stats: Time taken: 40.988 seconds, Loss value: 1151.915
Model 26 stats: Time taken: 26.474 seconds, Loss value: 923.730
Model 27 stats: Time taken: 32.037 seconds, Loss value: 904.503
Model 28 stats: Time taken: 33.611 seconds, Loss value: 905.850
Model 29 stats: Time taken: 156.003 seconds, Loss value: 902.330
Model 30 stats: Time taken: 88.890 seconds, Loss value: 954.624
'''
