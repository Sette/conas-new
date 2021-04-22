import os
import sys
import glob
import numpy as np
import time
import torch
import utils
import pickle
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.autograd import Variable



path_logits = "logits"
path_logits_darts = path_logits + "logits_intel"
path_logits1 = path_logits+"/logits1"
path_logits2 = path_logits+"/logits2"
path_logits3 = path_logits+"/logits3"
path_dataset = "data/intel/"


import pickle
import os
import sys
import glob
import json
import numpy as np
import torch
import utils
import logging
import argparse
import pickle
import time
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

CLASSES = 6 

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


save = path_logits +'/test-{}-{}'.format("ENS", time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
criterion_smooth = CrossEntropyLabelSmooth(CLASSES, 0.1)
criterion_smooth = criterion_smooth.cuda()


import torchvision.transforms as transforms
import torch
import torchvision.datasets as dset
import os
test_dir = os.path.join(path_dataset,'test')
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


objs = utils.AvgrageMeter()
top1 = utils.AvgrageMeter()
top5 = utils.AvgrageMeter()
x = 0
logits1 = pickle.load( open( path_logits1 + "/logits"+str(x)+".p", "rb" ) )
for step, (input, target) in enumerate(test_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)
    
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)
    x+=1

'''
for x in range(23):

  logits_darts = pickle.load( open( path_logits_darts + "/logits"+str(x)+".p", "rb" ) )
  print(type(logits_darts))
  logits1 = pickle.load( open( path_logits1 + "/logits"+str(x)+".p", "rb" ) )
  logits2 = pickle.load( open( path_logits2 + "/logits"+str(x)+".p", "rb" ) )
  logits3 = pickle.load( open( path_logits3 + "/logits"+str(x)+".p", "rb" ) )
  logits = copy.deepcopy(logits1)

  for i in range(len(logits)):
    for j in range(len(logits[i])):
      logits[i][j] = logits[i][j] + logits2[i][j] + logits3[i][j]

  for step, (input, target) in enumerate(test_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)
    
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

'''