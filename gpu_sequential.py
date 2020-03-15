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
        pipeline_fn, models = bench_suite[data_pipeline]
        for model in models:
            time_to_train = now()
            num_models_trained += 1
            # Train this model

            net = model()
            #net.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            for epoch in range(num_epochs):  # loop over the dataset multiple times

                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data

                    #NOTE: What does this line at the bottom do/whats the difference between line 64 and line 61?
                    inputs = torch.stack([pipeline_fn(inputs[i]) for i in range(inputs.shape[0])])

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

Trained models with an average throughput of 18.883078
Trained models at an average time of 52.957 seconds
Model 1 stats: Time taken: 19.366 seconds, Loss value: 947.540
Model 2 stats: Time taken: 20.304 seconds, Loss value: 762.926
Model 3 stats: Time taken: 19.453 seconds, Loss value: 822.485
Model 4 stats: Time taken: 26.359 seconds, Loss value: 718.396
Model 5 stats: Time taken: 29.558 seconds, Loss value: 758.594
Model 6 stats: Time taken: 31.722 seconds, Loss value: 829.797
Model 7 stats: Time taken: 46.052 seconds, Loss value: 861.724
Model 8 stats: Time taken: 47.935 seconds, Loss value: 944.168
Model 9 stats: Time taken: 50.229 seconds, Loss value: 1151.996
Model 10 stats: Time taken: 51.695 seconds, Loss value: 1151.478
Model 11 stats: Time taken: 22.842 seconds, Loss value: 672.778
Model 12 stats: Time taken: 35.997 seconds, Loss value: 637.372
Model 13 stats: Time taken: 37.604 seconds, Loss value: 658.133
Model 14 stats: Time taken: 162.197 seconds, Loss value: 585.431
Model 15 stats: Time taken: 101.224 seconds, Loss value: 611.642
Model 16 stats: Time taken: 27.235 seconds, Loss value: 1057.147
Model 17 stats: Time taken: 28.738 seconds, Loss value: 949.245
Model 18 stats: Time taken: 27.951 seconds, Loss value: 996.854
Model 19 stats: Time taken: 36.632 seconds, Loss value: 982.970
Model 20 stats: Time taken: 38.904 seconds, Loss value: 1042.490
Model 21 stats: Time taken: 40.789 seconds, Loss value: 1127.953
Model 22 stats: Time taken: 57.087 seconds, Loss value: 1152.112
Model 23 stats: Time taken: 58.189 seconds, Loss value: 1151.836
Model 24 stats: Time taken: 60.262 seconds, Loss value: 1151.348
Model 25 stats: Time taken: 61.717 seconds, Loss value: 1151.613
Model 26 stats: Time taken: 32.343 seconds, Loss value: 932.099
Model 27 stats: Time taken: 49.035 seconds, Loss value: 886.731
Model 28 stats: Time taken: 50.498 seconds, Loss value: 907.535
Model 29 stats: Time taken: 197.464 seconds, Loss value: 901.640
Model 30 stats: Time taken: 119.341 seconds, Loss value: 945.179



Hardware: Intel(R) Core(TM) i7-9700F CPU @ 3.00 GHz 3.00 GHz

Time for model:  88890
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
