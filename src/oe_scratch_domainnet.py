import argparse

import fastcore.all as fc
from collections.abc import Mapping
from pathlib import Path
from operator import attrgetter,itemgetter
from functools import partial
from copy import copy
from contextlib import contextmanager
from fastai.callback.all import SaveModelCallback
from fastai.callback.core import Callback
import math
from fastai.vision.all import *
from fastai.losses import BaseLoss

import neptune
from neptune.integrations.fastai import NeptuneCallback

from fastai.callback.schedule import lr_find, SchedCos, ParamScheduler

from getpass import getpass

import sys

import torchvision.transforms.functional as TF,torch.nn.functional as F
from torch import tensor,nn,optim
from torch.utils.data import DataLoader,default_collate
from torch.nn import init
from datasets import load_dataset,load_dataset_builder

#from fastai_functions.neptune_callback import NeptuneCallback
from fastai_functions.fastai_callbacks import CustomTrackLearningRate
from fastai.data.load import _FakeLoader, _loaders

import torchvision.transforms as trn
import torchvision.datasets as dset

from utils.validation_dataset import validation_split
from utils.calibration_tools import *
from utils.soft_binned_ece import *
from utils.data_prep_functions import *

from models_code.allconv import AllConvNet
from models_code.allconvdeconf import AllConvNetDeConf

parser = argparse.ArgumentParser(description='Train a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--use_01', '-z', action='store_true', help='Use 0-1 Posterior Rescaling.')
# WRN Architecture
parser.add_argument('--arch', type='str', default='AllConvNet', help='Define model to use')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/oe_tune', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='./snapshots/baseline', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

#torch.cuda.set_device(2)

# Open the .txt file
with open('./data/label.labels.txt', 'r') as file:
    # Read the content of the file
    content = file.readlines()

# Strip whitespace and newlines from each line and create a list
lines = [line.strip() for line in content]
cat = [i for i , _ in enumerate(lines)]


# loding .npz and .npy files for getting domainnet dataset.

domainnet_real_train_images = np.load('./data/domainnet_real_train_images.npz')
domainnet_real_train_labels = np.load('./data/domainnet_real_train_labels.npy')

domainnet_real_test_images = np.load('./data/domainnet_real_test_images.npz')
domainnet_real_test_labels = np.load('./data/domainnet_real_test_labels.npy')

domainnet_infograph_train_images = np.load('./data/domainnet_infograph_train_images.npz')
domainnet_infograph_train_labels = np.load('./data/domainnet_infograph_train_labels.npy')

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]


dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock(vocab = cat)),
    get_items=lambda x: x,
    splitter=RandomSplitter(valid_pct = 0.1, seed=42),
    get_x = lambda x: x['images'],
    get_y=lambda x: x['labels'],
    item_tfms=[Resize(32)],
    batch_tfms=[Normalize.from_stats(mean, std)]
)

train_image_dict_list = return_pair_dict_domainnet(domainnet_real_train_images, domainnet_real_train_labels)
train_dls = dblock.dataloaders(train_image_dict_list, bs = args.batch_size, shuffle=True,
    num_workers=4, pin_memory=True)

def nosplit(o): return L(int(i) for i in range(len(o))), L()

dblock_test = DataBlock(
    blocks=(ImageBlock, CategoryBlock(vocab = cat)),
    get_items=lambda x: x,
    splitter=nosplit,
    get_x = lambda x: x['images'],
    get_y=lambda x: x['labels'],
    item_tfms=[Resize(32)],
    batch_tfms=[Normalize.from_stats(mean, std)]
)

test_image_dict_list = return_pair_dict_domainnet(domainnet_real_test_images, domainnet_real_test_labels)
test_dls = dblock_test.dataloaders(test_image_dict_list, bs = args.test_bs, shuffle=True,
    num_workers=4, pin_memory=True)


dblock_ood = DataBlock(
    blocks=(ImageBlock, CategoryBlock(vocab = cat)),
    get_items=lambda x: x,
    splitter=RandomSplitter(valid_pct = 0.5, seed=42),
    get_x = lambda x: x['images'],
    get_y=lambda x: x['labels'],
    item_tfms=[Resize(32)],
    batch_tfms=[Normalize.from_stats(mean, std)]
)

# for ood data
ood_image_dict_list = return_pair_dict_domainnet(domainnet_infograph_train_images, domainnet_infograph_train_labels)
ood_dls = dblock_ood.dataloaders(ood_image_dict_list, bs = args.oe_batch_size, shuffle = True, num_workers=4,
                                  pin_memory = True)

# defining model
num_classes = len(cat)
if args.arch == 'AllConvNet':
    net = AllConvNet(num_classes=num_classes)
elif args.arch == 'AllConvNetDeConf':
    net = AllConvNetDeConf(num_classes=num_classes)
else:
    print("Please provide a correct architecture ['AllConvNet' or 'AllConvNetDeConf']")

train_mixed = MixedDL(train_dls[0], ood_dls[1])

#valid_mixed = MixedDL(train_loader[1])

dls_mixed = DataLoaders(train_mixed, train_dls[1])
#print(dls_mixed.device)
# for fixing neptunr error
dls_mixed.bs = train_dls.bs + ood_dls.bs
dls_mixed.n = train_dls.n + ood_dls.n


# defining custom loss function 
class CustomLoss(torch.nn.Module):
    def forward(self, x, target):
        target = target.long()
        loss = F.cross_entropy(x[:len(target)], target)
        # cross-entropy from softmax distribution to uniform distribution (calculating loss for ood Data)
        if len(target) > x.shape[0]:
            loss += 0.5 * -(x[len(target):].mean(1) - torch.logsumexp(x[len(target):], dim=1)).mean()
        return loss

# Create Neptune run
run = neptune.init_run(project='project_name', mode = 'debug', name='oe_scratch_domainnet_allconvnet',
                         api_token = "API Token")

# Create Neptune callback
neptune_callback = NeptuneCallback(run=run)

def SGD_with_momentum(params, lr=args.learning_rate, mom=0.9, weight_decay=0.005):
    return SGD(params, lr=lr, mom=mom, wd=weight_decay, decouple_wd=True)  # True weight decay

learn_new = Learner(dls_mixed, net, loss_func=CustomLoss(), lr = args.learning_rate, metrics=accuracy, opt_func=SGD_with_momentum, cbs=[CustomTrackLearningRate()])


learn_new.lr_find()

# min and max learning_rate
lr_min = 1e-6 / args.learning_rate
lr_max = 1

# Set up the scheduler
sched = {'lr': SchedCos(lr_max, lr_min)}
lr_scheduler = ParamScheduler(sched)


learn_new.fit(10, cbs=[lr_scheduler, 
                   SaveModelCallback(monitor='valid_loss', min_delta=2e-3, fname='domainnet_oe_scratch_allconvnet', every_epoch=False, with_opt=True),
                           neptune_callback])

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.to('cpu').numpy()


def get_net_results(data_loader, in_dist=False, t=1, valid = False):
    logits = []
    confidence = []
    correct = []
    labels = []

    if valid:
      data = data_loader.valid
    else: 
      data = data_loader.train

    output, target = learn_new.get_preds(dl = data)
    
    logits.extend(to_np(output).squeeze())

    if args.use_01:
        confidence.extend(to_np(
            (F.softmax(output/t, dim=1).max(1)[0] - 1./num_classes)/(1 - 1./num_classes)
        ).squeeze().tolist())
    else:
        confidence.extend(to_np(F.softmax(output/t, dim=1).max(1)[0]).squeeze().tolist())

    if in_dist:
        pred = output.data.max(1)[1]
        correct.extend(pred.eq(target).to('cpu').numpy().squeeze().tolist())
        labels.extend(target.tolist())

    if in_dist:
        return logits.copy(), confidence.copy(), correct.copy(), labels.copy()
    else:
        return logits.copy(), confidence.copy()
    

#for validation set
val_logits, val_confidence, val_correct, val_labels = get_net_results(train_dls, valid = True, in_dist = True)

# calculate temperature parameter on validation set
t_star = tune_temp(val_logits, val_labels)

#for test set
test_logits, test_confidence, test_correct, test_labels = get_net_results(test_dls, in_dist=True, t=t_star)

rms, mad, soft_f1_score = return_calibration_results(np.array(test_confidence), np.array(test_correct), method_name='df')

sb_ece = compute_squared_error_label_binning_pytorch(logits = test_logits, y = test_labels, m = 100, temperature = t_star)

# logging calibration error to neptune.ai
run['metrics/calib_error/tuned_temp'] = t_star
run['metrics/calib_error/rms'] = rms
run['metrics/calib_error/mad'] = mad
run['metrics/calib_error/soft_f1'] = soft_f1_score
run['metrics/calib_error/sb_ece'] = sb_ece

### for OOD sample ###

domainnet_infograph_test_images = np.load('./data/domainnet_infograph_test_images.npz')
domainnet_infograph_test_labels = np.load('./data/domainnet_infograph_test_labels.npy')


# Function to filter out items with these labels
def filter_include(items):
    return [item for item in items if item['labels'] in excluded_labels]

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock(vocab = cat)),
    get_items=lambda x: x,
    splitter=RandomSplitter(valid_pct = 0.3, seed=42),
    get_x = lambda x: x['images'],
    get_y=lambda x: x['labels'],
    item_tfms=[Resize(32)],
    batch_tfms=[Normalize.from_stats(mean, std)]
)
ood_list = return_pair_dict_domainnet(domainnet_infograph_test_images, domainnet_infograph_test_labels)
ood_dls = dblock.dataloaders(ood_list, bs = args.test_bs, shuffle=True,
    num_workers=4, pin_memory=True)

# /////////////// OOD Calibration ///////////////
rms_list, mad_list, sf1_list = [], [], []

def get_and_print_results(ood_loader, num_to_avg=1):

    rmss, mads, sf1s = [], [], []
    for _ in range(num_to_avg):
        out_logits, out_confidence = get_net_results(ood_loader, valid=True, t=t_star)
        #out_confidence = out_confidence[:2500]
        measures = get_measures(
            concat([out_confidence, test_confidence]),
            concat([np.zeros(len(out_confidence)), test_correct]))

        rmss.append(measures[0]); mads.append(measures[1]); sf1s.append(measures[2])

    rms = np.mean(rmss); mad = np.mean(mads); sf1 = np.mean(sf1s)
    rms_list.append(rms); mad_list.append(mad); sf1_list.append(sf1)
 
    if num_to_avg >= 5:
        print_measures_with_std(rmss, mads, sf1s, 'bh')
    else:
        return rms, mad, sf1, out_confidence, out_logits
    

rms_ood, mad_ood, sf1_ood, out_confidence, out_logits =  get_and_print_results(ood_dls)

# sb_ece for ood samples
output_ood, target_ood = learn_new.get_preds(dl = ood_dls.valid)
sb_ece_ood = compute_squared_error_label_binning_pytorch(logits = output_ood, y = target_ood, m = 100, temperature = t_star)

run['metrics/calib_error/rms_ood'] = rms_ood
run['metrics/calib_error/mad_ood'] = mad_ood
run['metrics/calib_error/soft_f1_ood'] = sf1_ood
run['metrics/calib_error/sb_ece_ood'] = sb_ece_ood

run.stop()
