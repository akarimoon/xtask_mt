import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler

from create_dataset import *
from utils import *
from pcgrad import *
from segnets import *

parser = argparse.ArgumentParser(description='Multi-task: Attention Network (inference time)')
parser.add_argument('--data', default='cs', type=str, choices=['cs', 'nyu'], help='dataset')
parser.add_argument('-n', '--num_tasks', type=int, default=2, help='number of tasks, 2 or 3')
opt = parser.parse_args()

num_tasks = opt.num_tasks
# define model, optimiser and scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if opt.data == 'cs':
    SegNet_MTAN = SegNet2tasks(class_nb=7).to(device)
elif opt.data == 'nyu':
    if num_tasks == 3:
        SegNet_MTAN = SegNet3tasks().to(device)
    else:
        SegNet_MTAN = SegNet2tasks(class_nb=13).to(device)

print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(SegNet_MTAN),
                                                         count_parameters(SegNet_MTAN) / 24981069))
# print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')

# define dataset
if opt.data == 'cs':
    dataset_path = '../../data/cityscapes'
    test_set = MyCityscapesDataset(height=128, width=256, root_path=dataset_path, num_classes=7, split='val',
                                      transform=None, ignore_index=250)
elif opt.data == 'nyu':
    dataset_path = '../../data/nyu'
    test_set = NYUv2(root=dataset_path, train=False)
print("# of tasks: {}".format(num_tasks))

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=1,
    shuffle=False)

# Train and evaluate multi-task network
infer_only(test_loader,
           SegNet_MTAN,
           device,
           opt,
           10, is_cs=opt.data == 'cs', num_tasks=num_tasks)
