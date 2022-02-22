import numpy as np
import time
import torch
import torchvision as vision
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utils import continuous_learning_rate, piecewise_learning_rate
from torch.autograd import Variable
import itertools as iter


training_transformations = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.RandomVerticalFlip(p=0.5),
                                               transforms.ColorJitter(brightness=0.3, contrast=0, saturation=0, hue=0),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
training_data = vision.datasets.CIFAR10(root='./CIFARdata', train=True, download=True, transform=training_transformations)
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=100, shuffle=True, num_workers=0)

testing_transformations = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testing_data = vision.datasets.CIFAR10(root='./CIFARdata', train=False, download=True, transform=testing_transformations)
testing_dataloader = torch.utils.data.DataLoader(testing_data, batch_size=100, shuffle=False, num_workers=0)

class CIFAR10Model(nn.Module):
    def __init__(self, num_outputs, dropout_prob=.5, channels=64):
        super(CIFAR10Model, self).__init__()

        '''
        No. of output units in a convolution layer = 
        {[No. of input units - Filter Size + 2(Padding)]/Stride} + 1 = 
        (32 - 3 + 2(1))/1 + 1 = 32 
        '''

        ###Convolution Layers
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=4, stride=1, padding=2)
        self.conv_layer2 = nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=4, stride=1, padding=2)
        self.conv_layer3 = nn.Conv2d(in_channels=2*channels, out_channels=channels, kernel_size=4, stride=1, padding=2)
        self.conv_layer4 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=1, padding=2)
        self.conv_layer5 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=1, padding=2)
        self.conv_layer6 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=0)
        self.conv_layer7 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=0)
        self.conv_layer8 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=0)

        ###Fully Connected Layers
        # Image size is 24*24 (due to random resized cropping) with 128 channels from the last convolution layer
        self.fc1 = nn.Linear(in_features=16*channels, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=num_outputs)

        ###Dropout Functions
        self.dropout = nn.Dropout(p=dropout_prob)  # p - Probability of dropping out a neuron
        self.dropout2d = nn.Dropout2d(p=dropout_prob) #Dropout2d zeros out an entire channel

        ###Batch Normalization and Pooling
        #The numbers here refer to the number of batch normalizations already done, not the convolution layer being batch normalized
        self.batchnorm2d_layer1 = nn.BatchNorm2d(channels) #Batch Normalization
        self.batchnorm2d_layer2 = nn.BatchNorm2d(channels)
        self.batchnorm2d_layer3 = nn.BatchNorm2d(channels)
        self.batchnorm2d_layer4 = nn.BatchNorm2d(channels)
        self.batchnorm2d_layer5 = nn.BatchNorm2d(channels)
        self.pool = nn.MaxPool2d(2, stride=2) #Max pooling

    def forward(self, x):  # Specifying the NN architecture
        x = F.relu(self.conv_layer1(x))  # Convolution layers with relu activation
        x = self.batchnorm2d_layer1(x)  # Batch normalization

        x = F.relu(self.conv_layer2(x))
        x = self.pool(x)  # Pooling layer
        x = self.dropout2d(x)  # Dropout

        x = F.relu(self.conv_layer3(x))
        x = self.batchnorm2d_layer2(x)
        x = F.relu(self.conv_layer4(x))
        x = self.pool(x)

        # x = self.dropout2d(x)
        x = F.relu(self.conv_layer5(x))
        x = self.batchnorm2d_layer3(x)
        x = F.relu(self.conv_layer6(x))

        # x = self.dropout2d(x)
        x = F.relu(self.conv_layer7(x))
        x = self.batchnorm2d_layer4(x)
        x = F.relu(self.conv_layer8(x))

        x = self.batchnorm2d_layer5(x)
        x = self.dropout2d(x)

        x = x.view(-1, 16*channels)  # Flattening the conv2D output for dropout
        x = F.relu(self.fc1(x))  # Fully connected layer with relu activation
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

num_outputs = 10

'''
Train Mode
Create the batch -> Zero the gradients -> Forward Propagation -> Calculating the loss 
-> Backpropagation -> Optimizer updating the parameters -> Prediction 
'''

param_tuples = iter.product([.25, .5, .75], [32, 64, 128], [.001, .01, .1])
results = []
#[[(0.5, 64, 0.01), 1675.6082527637482, 62.0]]
#[[(0.25, 32, 0.01), 1143.3575930595398, 63.0]]
#[(0.25, 32, 0.1), 1135.8746416568756, 13.0]

for params in param_tuples:
    dropout_prob = params[0]
    channels = params[1]
    LR_magnitude = params[2]

    model = CIFAR10Model(num_outputs, dropout_prob, channels)
    model.cuda()  # Sending the model to the GPU
    batch_size = 100  # Batch size
    loss_func = nn.CrossEntropyLoss()  # Cross entropy loss function
    model.train()
    train_accuracy = []

    start_time = time.time()
    for epoch in range(40):  # loop over the dataset multiple times
        print(epoch)
        #Defining the learning rate based on the no. of epochs
        LR = LR_magnitude * piecewise_learning_rate(epoch, c_1=5)
        optimizer = optim.Adam(model.parameters(), lr=LR) #ADAM optimizer
        running_loss = 0.0
        for i, batch in enumerate(training_dataloader, 0):
            data, target = batch
            data, target = Variable(data).cuda(), Variable(target).cuda()
            optimizer.zero_grad()  # Zero the gradients at each epoch
            output = model(data)  # Forward propagation
            # Negative Log Likelihood Objective function
            loss = loss_func(output, target)
            loss.backward()  # Backpropagation
            optimizer.step()  # Updating the parameters using ADAM optimizer
            prediction = output.data.max(1)[1]  # Label Prediction
            accuracy = (float(prediction.eq(target.data).sum()) / float(
                batch_size)) * 100.0  # Computing the training accuracy
            train_accuracy.append(accuracy)
        accuracy_epoch = np.mean(train_accuracy)
        print('\nIn epoch ', epoch, ' the accuracy of the training set =', accuracy_epoch)
    end_time = time.time()

    for batch in testing_dataloader:
        test_accuracy = []
        data, target = batch
        data, target = Variable(data).cuda(), Variable(target).cuda()
        output = model(data) # Averaging the softmmax probabilities montecarlo no. of times
        prediction = output.data.max(1)[1]  # Label Prediction
        accuracy = (float(prediction.eq(target.data).sum()) / float(batch_size)) * 100.0  # Computing the test accuracy
        test_accuracy.append(accuracy)

    accuracy_test2 = np.mean(test_accuracy)
    print('\nAccuracy on the test set = ', accuracy_test2)

    results.append([params, end_time - start_time, accuracy_test2])
    print(results)

with open('results.txt', 'w') as f:
    f.write(str(results))