import torch
import torchvision
import torchvision.transforms as tfs
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datasets

import data
import model
import EZplot

num_epochs = 1
#batch_size_train = 64
#batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5

train_dict, test_dict = data.prep_data()

network = model.Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)


train_losses = []
test_losses = {}
for subtest in test_dict.keys():
    test_losses[subtest] = []
test_correct = {}
for subtest in test_dict.keys():
    test_correct[subtest] = []


for subtest, test_loader in test_dict.items():
    model.test(network, test_loader, test_losses[subtest], test_correct[subtest], subtest)

for task, train_loader in train_dict.items():
    for epoch in range(1, num_epochs + 1):
        model.train(network, train_loader, train_losses, optimizer, epoch)
        for subtest, test_loader in test_dict.items():
            model.test(network, test_loader, test_losses[subtest], test_correct[subtest], subtest)


EZplot.plot_single(train_losses, "train loss", filename="train_loss.png")
EZplot.plot_multi(
    list(test_losses.values()), 
    [subtest+" loss" for subtest in test_losses.keys()],
    ['k', 'r', 'm', 'b', 'g', 'y'],
    filename="test_losses.png")
EZplot.plot_multi(
    list(test_correct.values()), 
    ["jó predikciók - "+subtest for subtest in test_losses.keys()], 
    ['k', 'r', 'm', 'b', 'g', 'y'],
    filename="test_correct.png")