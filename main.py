import torch
import torchvision
import torchvision.transforms as tfs
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datasets
import copy

import data
import model
#import EZplot
import mutinfo

num_epochs = 1
#batch_size_train = 64
#batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
sample_size = 4096

train_dict, test_dict = data.prep_data()

network_0 = model.Net()
optimizer = optim.SGD(network_0.parameters(), lr=learning_rate, momentum=momentum)

model.train(network_0, train_dict['task_0'], [], optimizer, 1)

results_dict = {}


#ezt szebb lett volna függvénnyel paraméteresen meghívni
network_1 = copy.deepcopy(network_0)
model.train(network_1, train_dict['task_1'], [], optimizer, 1)
results_dict['M_01, M_01->23 | D_01'] = mutinfo.mutual_information(
    network_0, network_1, sample_size, test_dict['task_0'], 1)
results_dict['M_01, M_01->23 | D_23'] = mutinfo.mutual_information(
    network_0, network_1, sample_size, test_dict['task_1'], 1)
results_dict['M_01, M_01->23 | D_0123'] = mutinfo.mutual_information(
    network_0, network_1, sample_size, test_dict['task_2'], 1)

network_2 = model.Net()
model.train(network_2, train_dict['task_2'], [], optimizer, 2)
results_dict['M_01, M_0123 | D_01'] = mutinfo.mutual_information(
    network_0, network_2, sample_size, test_dict['task_0'], 1)
results_dict['M_01, M_0123 | D_23'] = mutinfo.mutual_information(
    network_0, network_2, sample_size, test_dict['task_1'], 1)
results_dict['M_01, M_0123 | D_0123'] = mutinfo.mutual_information(
    network_0, network_2, sample_size, test_dict['task_2'], 1)

network_3 = copy.deepcopy(network_0)
model.train(network_3, train_dict['task_3'], [], optimizer, 3)
results_dict['M_01, M01->67 | D_01'] = mutinfo.mutual_information(
    network_0, network_3, sample_size, test_dict['task_0'], 1)
results_dict['M_01, M01->67 | D_67'] = mutinfo.mutual_information(
    network_0, network_3, sample_size, test_dict['task_3'], 1)
results_dict['M_01, M01->67 | D_0167'] = mutinfo.mutual_information(
    network_0, network_3, sample_size, test_dict['task_4'], 1)

print(results_dict)
#for subtest, test_loader in test_dict.items():
#    model.test(network, test_loader, test_losses[subtest], test_correct[subtest], subtest)

#EZplot.plot_single(train_losses, "train loss")#, filename="train_loss.png")
#EZplot.plot_multi(
#    list(test_losses.values()), 
#    [subtest+" loss" for subtest in test_losses.keys()],
#    ['k', 'r', 'm', 'b', 'g', 'y'],
#    #filename="test_losses.png"
#    )