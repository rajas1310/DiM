import torch
import torch.nn as nn
from torchvision.models import models
import numpy as np


# def get_resnet10(num_classes):
#   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#   net = models.resnet10(pretrained=True)
#   net = net.cuda() if device else net
#   num_ftrs = net.fc.in_features
#   net.fc = nn.Linear(num_ftrs, num_classes)
#   net.fc = net.fc.cuda() if device else net.fc
#   return net

def get_resnet18(num_classes):  # 11.7 M parameters 
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  net = models.resnet18(pretrained=True)
  net = net.cuda() if device else net
  num_ftrs = net.fc.in_features
  net.fc = nn.Linear(num_ftrs, num_classes)
  net.fc = net.fc.cuda() if device else net.fc
  return net

def get_resnet34(num_classes):  # 22 M parameters 
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  net = models.resnet34(pretrained=True)
  net = net.cuda() if device else net
  num_ftrs = net.fc.in_features
  net.fc = nn.Linear(num_ftrs, num_classes)
  net.fc = net.fc.cuda() if device else net.fc
  return net

def get_efficientnetB0(num_classes):  # 5.2 M parameters
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  net = models.efficientnet_b0(pretrained=True)
  net = net.cuda() if device else net
  num_ftrs = net.fc.in_features
  net.fc = nn.Linear(num_ftrs, num_classes)
  net.fc = net.fc.cuda() if device else net.fc
  return net

def get_efficientnetB1(num_classes): # 7.8 M params
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  net = models.efficientnet_b1(pretrained=True)
  net = net.cuda() if device else net
  num_ftrs = net.fc.in_features
  net.fc = nn.Linear(num_ftrs, num_classes)
  net.fc = net.fc.cuda() if device else net.fc
  return net
  
def get_alexnet(num_classes): # 61 M parameters 
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  net = models.alexnet(pretrained=True)
  net = net.cuda() if device else net
  num_ftrs = net.fc.in_features
  net.fc = nn.Linear(num_ftrs, num_classes)
  net.fc = net.fc.cuda() if device else net.fc
  return net

def get_convnextTiny(num_classes): # 28.6 M params
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  net = models.convnext_tiny(pretrained=True)
  net = net.cuda() if device else net
  num_ftrs = net.fc.in_features
  net.fc = nn.Linear(num_ftrs, num_classes)
  net.fc = net.fc.cuda() if device else net.fc
  return net

def get_efficientnetV2_s(num_classes): #21.4 M parameters
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  net = models.efficientnet_v2_s(pretrained=True)
  net = net.cuda() if device else net
  num_ftrs = net.fc.in_features
  net.fc = nn.Linear(num_ftrs, num_classes)
  net.fc = net.fc.cuda() if device else net.fc
  return net

