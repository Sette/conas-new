import pickle
import torch
import os
import pickle
import sys
import numpy as np
from torch.autograd import Variable
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss	

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
criterion_smooth = CrossEntropyLabelSmooth(10, 0.1)
criterion_smooth = criterion_smooth.cuda()


path_logits = "logits"
path_logits_darts = path_logits + "logits_intel"
path_logits1 = path_logits+"/logits1"
path_logits2 = path_logits+"/logits2"
path_logits3 = path_logits+"/logits3"
path_dataset = "data/intel/"

test_dir = os.path.join("data/intel", 'test')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_data = dset.ImageFolder(
    test_dir,
    transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
      transforms.ToTensor(),
      normalize,
    ]))

test_queue = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, pin_memory=True, num_workers=2)

'''
logits1 = pickle.load( open( path_logits1 + "/logits0.p", "rb" ) )
for i in range(23):
	logits1 = pickle.load( open( path_logits1 + "/logits"+str(i)+".p", "rb" ) )
	logits2 = pickle.load( open( path_logits2 + "/logits"+str(i)+".p", "rb" ) )
	logits3 = pickle.load( open( path_logits3 + "/logits"+str(i)+".p", "rb" ) )
	logits = torch.add(logits1,logits2)
	logits = torch.add(logits,logits3)
	print(logits[1])
'''
objs = utils.AvgrageMeter()
top1 = utils.AvgrageMeter()
top5 = utils.AvgrageMeter()
x = 0
for step, (input, target) in enumerate(test_queue):
	input = Variable(input, volatile=True).cuda()
	target = Variable(target, volatile=True).cuda(async=True)
	logits1 = pickle.load( open( path_logits1 + "/logits"+str(x)+".p", "rb" ) )
	logits2 = pickle.load( open( path_logits2 + "/logits"+str(x)+".p", "rb" ) )
	logits3 = pickle.load( open( path_logits3 + "/logits"+str(x)+".p", "rb" ) )
	logits = torch.add(logits1,logits2)
	logits = torch.add(logits,logits3)
	print(logits[1])
	loss = criterion(logits, target)
	prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
	n = input.size(0)
	objs.update(loss.data[0], n)
	top1.update(prec1.data[0], n)
	top5.update(prec5.data[0], n)


