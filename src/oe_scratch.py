import argparse

from fastai.callback.all import SaveModelCallback
from fastai.callback.core import Callback
import math
from fastai.vision.all import *

import neptune
from neptune.integrations.fastai import NeptuneCallback

from fastai.callback.schedule import lr_find, SchedCos, ParamScheduler
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
parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--use_01', '-z', action='store_true', help='Use 0-1 Posterior Rescaling.')
# WRN Architecture
parser.add_argument('--arch', type='str', default='AllConvNet', help='Define model to use')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
#parser.add_argument('-f')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)


torch.manual_seed(1)
np.random.seed(1)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])


train_data = dset.CIFAR10('./data', train=True, transform=train_transform, download=True)
test_data = dset.CIFAR10('./data', train=False, transform=test_transform, download = True)
num_classes = 10


# creating train and test loader
train_loader = make_dataloaders_from_numpy_data(train_data.data, train_data.targets, batch_size=128)
test_loader = make_dataloaders_from_numpy_data(test_data.data, test_data.targets, batch_size=200, train=False)

ood_data = NumpyDataset("./data/300K_random_images.npy", transform=train_transform)

train_loader_ood = make_dataloaders_from_numpy_data(ood_data.data, batch_size = 256, train=False, ood=True)


train_mixed = MixedDL(train_loader[0], train_loader_ood[0])
#valid_mixed = MixedDL(train_loader[1])

dls_mixed = DataLoaders(train_mixed, train_loader[1])

# for fixing neptunr error
dls_mixed.bs = 384
dls_mixed.n = 340000

# defining model
if args.arch == 'AllConvNet':
    net = AllConvNet(num_classes=num_classes)
elif args.arch == 'AllConvNetDeConf':
    net = AllConvNetDeConf(num_classes=num_classes)
else:
    print("Please provide a correct architecture ['AllConvNet' or 'AllConvNetDeConf']")

def SGD_with_momentum(params, lr=0.1, mom=0.9, weight_decay=0.005):
    return SGD(params, lr=lr, mom=mom, wd=weight_decay, decouple_wd=True)  # True weight decay

# defining custom loss function 
class CustomLoss(torch.nn.Module):
    def forward(self, x, target):
        target = target.long()
        loss = F.cross_entropy(x[:len(target)], target)
        # cross-entropy from softmax distribution to uniform distribution (calculating loss for ood Data)
        if len(target) > x.shape[0]:
            loss += 0.5 * -(x[len(target):].mean(1) - torch.logsumexp(x[len(target):], dim=1)).mean()
        return loss

# defining learner
learn = Learner(dls_mixed, net, loss_func=CustomLoss(), lr = args.learning_rate, opt_func=SGD_with_momentum, metrics=accuracy, cbs=CustomTrackLearningRate())

# Find a suitable learning rate using lr_find
learn.lr_find()

# min and max learning_rate
lr_min = 1e-6 / args.learning_rate
lr_max = 1

# Set up the scheduler
sched = {'lr': SchedCos(lr_max, lr_min)}
lr_scheduler = ParamScheduler(sched)

# Create Neptune run
run = neptune.init_run(project='project_name', mode = 'debug' name='traning_with_oe_deconf_arch',
                         api_token = "API Token")

# Create Neptune callback
neptune_callback = NeptuneCallback(run=run)

learn.fit(args.epochs, cbs=[lr_scheduler, 
                   SaveModelCallback(monitor='valid_loss', min_delta=2e-3, fname='cifar_10_oe_scratch_allconvnetdeconf', every_epoch=False, with_opt=True),
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

    output, target = learn.get_preds(dl = data)
    
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
    
#for validation set
val_logits, val_confidence, val_correct, val_labels = get_net_results(train_loader, valid = True, in_dist = True)

# calculate temperature parameter on validation set
t_star = tune_temp(val_logits, val_labels)

# before temperature tuning
test_logits, test_confidence, test_correct, test_labels = get_net_results(test_loader, in_dist=True, t=1)

rms, mad, soft_f1_score = return_calibration_results(np.array(test_confidence), np.array(test_correct), method_name='df')

run['metrics/calib_error/rms_before_ts'] = rms
run['metrics/calib_error/mad_before_ts'] = mad
run['metrics/calib_error/soft_f1_before_ts'] = soft_f1_score


#for test set
test_logits, test_confidence, test_correct, test_labels = get_net_results(test_loader, in_dist=True, t=t_star)

rms, mad, soft_f1_score = return_calibration_results(np.array(test_confidence), np.array(test_correct), method_name='df')

sb_ece = compute_squared_error_label_binning_pytorch(logits = test_logits, y = test_labels, m = 100, temperature = t_star)

# logging calibration error to neptune.ai
run['metrics/calib_error/tuned_temp'] = t_star
run['metrics/calib_error/rms'] = rms
run['metrics/calib_error/mad'] = mad
run['metrics/calib_error/soft_f1'] = soft_f1_score
run['metrics/calib_error/sb_ece'] = sb_ece

# /////////////// OOD Calibration ///////////////
rms_list, mad_list, sf1_list = [], [], []

def get_and_print_results(ood_loader, num_to_avg=1):

    rmss, mads, sf1s = [], [], []
    for _ in range(num_to_avg):
        out_logits, out_confidence = get_net_results(ood_loader, valid=True, t=t_star)
        out_confidence = out_confidence[:1500]
        measures = get_measures(
            concat([out_confidence, test_confidence]),
            concat([np.zeros(len(out_confidence)), test_correct]))

        rmss.append(measures[0]); mads.append(measures[1]); sf1s.append(measures[2])

    rms = np.mean(rmss); mad = np.mean(mads); sf1 = np.mean(sf1s)
    rms_list.append(rms); mad_list.append(mad); sf1_list.append(sf1)

    if num_to_avg >= 5:
        print_measures_with_std(rmss, mads, sf1s, 'bh')
    else:
        return rms, mad, sf1, out_confidence
    

rms_ood, mad_ood, sf1_ood, out_confidence =  get_and_print_results(train_loader_ood)

run['metrics/calib_error/rms_ood'] = rms_ood
run['metrics/calib_error/mad_ood'] = mad_ood
run['metrics/calib_error/soft_f1_ood'] = sf1_ood

run.stop()
