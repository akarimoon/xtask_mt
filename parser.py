import argparse

def nyu_parser():
    weights_loadable = ["./model.pt", "./weights_nyu.pth"]

    parser = argparse.ArgumentParser(description='Relabs of Midas to NYU Dataset')
    parser.add_argument('--load_weights', default='./MiDaS/model.pt', choices=weights_loadable,
                        help='model architecture: ' + ' | '.join(weights_loadable) + ' (default: Midas pretrained)')
    parser.add_argument('--input_path', default='./data/nyu/nyu_depth_v2_labeled.mat',
                        help='path of dataset (default: /data/nyu/nyu_depth_v2_labeled.mat)')
    parser.add_argument('--train_frac', type=float, default=0.8,
                        help='training size (fraction) (default: 0.8)')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='number of epochs (default: 10)')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('-n', '--sample_count', type=int, default=384*384,
                        help='sample count for sparse depth (default: 10000)')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--b1', '--beta_1', type=float, default=0.9,
                        help='beta_1 of Adam (default: 0.9)')
    parser.add_argument('--b2', '--beta_2', type=float, default=0.99,
                        help='beta_2 of Adam (default: 0.99)')
    parser.add_argument('-a', '--alpha', type=float, default=0.5,
                        help='alpha of loss function (default: 0.5)')
    parser.add_argument('--save_weights', default='./weights_nyu.pth',
                        help='path to where weights are saved (default: weights_nyu_relabs.pth)')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--progress', '--save_progress', default=1,
                        help='flag: save image per epoch (default: 1 (True))')
    args = parser.parse_args()

    return args


def cityscapes_xtask_parser():

    parser = argparse.ArgumentParser(description='XTask MT on Cityscapes Dataset')
    parser.add_argument('--input_path', default='../data/cityscapes',
                        help='path of dataset (default: ../data/cityscapes)')
    parser.add_argument('--height', type=int, default=256,
                        help='height of output (default: 256)')
    parser.add_argument('--width', type=int, default=512,
                        help='width of output (default: 512)')

    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='number of epochs (default: 50)')
    parser.add_argument('--enc_layers', type=int, default=34,
                        help='type of ResNet encoder (default: 34)')
    parser.add_argument('-b', '--batch_size', type=int, default=6,
                        help='batch size (default: 6)')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--b1', '--beta_1', type=float, default=0.9,
                        help='beta_1 of Adam (default: 0.9)')
    parser.add_argument('--b2', '--beta_2', type=float, default=0.99,
                        help='beta_2 of Adam (default: 0.99)')
    parser.add_argument('-n', '--num_classes', type=int, default=19, choices=[7, 19],
                        help='number of classes for segmentation task (default: 19)')
    parser.add_argument('--scheduler_step_size', type=int, default=15,
                        help='step size of scheduler (steplr)')
    parser.add_argument('--scheduler_gamma', default=0.1,
                        help='lr decay of scheduler (steplr)')

    parser.add_argument('-a', '--alpha', type=float, default=0.4,
                        help='alpha of loss function (default: 0.4)')
    parser.add_argument('-g', '--gamma', type=float, default=0.1,
                        help='gamma of loss function (default: 0.1')
    parser.add_argument('--label_smoothing', type=float, default=0.,
                        help='label smoothing when calculating KL loss')
    parser.add_argument('--lp', default="MSE", choices=["MSE", "L1", "logL1"],
                        help='depth loss for depth loss')
    parser.add_argument('--tseg_loss', default="cross", choices=["cross", "kl"],
                        help='label loss for cross-task segmt loss')

    parser.add_argument('--uncertainty_weights', action='store_true',
                        help='flag: use uncertainty weights (Kendall+, 2018) for balancing cross-task losses')
    # parser.add_argument('--pcgrad', action='store_true',
    #                     help='flag: use pc grad (Yu+, 2020) for cross-task losses')
    parser.add_argument("--grad_loss", action='store_true',
                        help='use grad loss')

    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--save_path', default='./tmp/',
                        help='path to folder where weights are saved (default: ./tmp/)')

    parser.add_argument('--infer_only', action='store_true', 
                        help='flag: only infer')
    parser.add_argument('--exp_num', type=int,
                        help='experiment number (only for infer_only)')
    parser.add_argument('--view_only', action='store_true', 
                        help='flag: do not save graphs')
    parser.add_argument('--run_only', action='store_true', 
                        help='flag: do not show graphs')
    parser.add_argument('--cpu', action='store_true', 
                        help='flag: use cpu')
    parser.add_argument('--debug', action='store_true', 
                        help='flag: debug mode, dont save weights')
    parser.add_argument('--notqdm', action='store_true',
                        help='flag: disable tqdm')

    args = parser.parse_args()

    return args