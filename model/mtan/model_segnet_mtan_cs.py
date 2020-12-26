import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler

from create_dataset import *
from utils import *
from pcgrad import *
from segnets import *

parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
parser.add_argument('--dataroot', default='cs', type=str, help='dataset root')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on CityScapes')
parser.add_argument('--pcgrad', action='store_true', help='toggle to use pcgrad')
parser.add_argument('-n', '--num_classes', default=7, type=int, choices=[7, 19])
opt = parser.parse_args()

# define model, optimiser and scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SegNet_MTAN = SegNet2tasks(class_nb=opt.num_classes).to(device)
if not opt.pcgrad:
    optimizer = optim.Adam(SegNet_MTAN.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
else:
    optimizer = PCGrad(optim.Adam(SegNet_MTAN.parameters(), lr=1e-4))
    scheduler = None
    print("Using PCGrad")

print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(SegNet_MTAN),
                                                         count_parameters(SegNet_MTAN) / 24981069))
# print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')

# define dataset
dataset_path = opt.dataroot
if opt.num_classes == 7:
    if opt.apply_augmentation:
        cs_train_set = MyCityscapesDataset(height=128, width=256, root_path=dataset_path, num_classes=7, split='train',
                                             transform=['random_flip', 'random_crop'], ignore_index=250)
        # cs_train_set = Cityscapes(root=dataset_path, train=True)
        print('Applying data augmentation on Cityscapes7.')
    else:
        cs_train_set = MyCityscapesDataset(height=128, width=256, root_path=dataset_path, num_classes=7, split='train',
                                             transform=None, ignore_index=250)
        # cs_train_set = Cityscapes(root=dataset_path, train=True)
        print('Standard training strategy without data augmentation on Cityscapes7.')
    cs_test_set = MyCityscapesDataset(height=128, width=256, root_path=dataset_path, num_classes=7, split='val',
                                             transform=None, ignore_index=250)
    # cs_test_set = Cityscapes(root=dataset_path, train=False)

elif opt.num_classes == 19:
    if opt.apply_augmentation:
        cs_train_set = MyCityscapesDataset(height=128, width=256, root_path=dataset_path, num_classes=19, split='train',
                                             transform=['random_flip'], ignore_index=-1)
        print('Applying data augmentation on Cityscapes19.')
    else:
        cs_train_set = MyCityscapesDataset(height=128, width=256, root_path=dataset_path, num_classes=19, split='train',
                                             transform=None, ignore_index=250)
        print('Standard training strategy without data augmentation on Cityscapes19.')
    cs_test_set = MyCityscapesDataset(height=128, width=256, root_path=dataset_path, num_classes=19, split='val',
                                             transform=None, ignore_index=250)

print("Task weighting method: {}".format(opt.weight))

batch_size = 2
cs_train_loader = torch.utils.data.DataLoader(
    dataset=cs_train_set,
    batch_size=batch_size,
    shuffle=True, num_workers=4)

cs_test_loader = torch.utils.data.DataLoader(
    dataset=cs_test_set,
    batch_size=batch_size if not opt.time_inf else 1,
    shuffle=False, num_workers=4)

# Train and evaluate multi-task network
multi_task_trainer(cs_train_loader,
                   cs_test_loader,
                   SegNet_MTAN,
                   device,
                   optimizer,
                   scheduler,
                   opt,
                   200,
                   is_cs=True,
                   num_tasks=2)

