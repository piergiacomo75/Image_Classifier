import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models


def NetworkProject(pretrained_model, 
                   classifier_parameters, 
                   p_dropout, 
                   learning_rate):
    
    # Freeze parameters so we don't backprop through them
    # Turn off gradient
    for param in pretrained_model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(classifier_parameters[0], classifier_parameters[1])),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(classifier_parameters[1], classifier_parameters[2])),
                          ('relu', nn.ReLU()),
                          ('drop_out', nn.Dropout(p=p_dropout)),    
                          ('fc3', nn.Linear(classifier_parameters[2], classifier_parameters[3])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    pretrained_model.classifier = classifier
    return pretrained_model


def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath)
    model = NetworkProject(model, 
                           checkpoint['parameters'],
                           checkpoint['p_dropout'],
                           checkpoint['learning_rate'])
    
    #model.load_state_dict(checkpoint['state_dict'])
    
    return model


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

