"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def Net(params):
    model = models.resnet18(pretrained=False)
    ## add final FC layer
    # model = torch.nn.Sequential(model, torch.nn.Linear(1000, 2))
    # model.ap = torch.nn.AdaptiveAvgPool2d(output_size=1)
    # model.bn1 = torch.nn.BatchNorm1d(1000, eps = 1e-05, momentum=.1, affine=True, track_running_stats=True)
    # model.do1 = torch.nn.Dropout(params.dropout_rate, inplace=True)
    num_features = model.fc.in_features
    num_feat2 = int(num_features / 2)
    model.fc = torch.nn.Linear(num_features, num_feat2)
    model.add_module("fc_rl1", torch.nn.ReLU(inplace = True))
    model.add_module("fc_bn1", torch.nn.BatchNorm2d(num_feat2))
    model.add_module("fc_do1", torch.nn.Dropout(params.dropout_rate, inplace=True))
    model.add_module("fc_fc2", torch.nn.Linear(num_feat2, 2))
    return model



def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    # num_examples = outputs.size()[0]
    # outputs = outputs[range(num_examples), labels]
    # return -torch.sum(outputs[range(num_examples), labels])/num_examples
    # outputs = outputs[:, 1]
    loss = nn.BCEWithLogitsLoss()
    target = labels.to(torch.float32)

    return loss(outputs, target)



def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
