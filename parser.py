import argparse

def nyu_xtask_parser():

    parser = argparse.ArgumentParser(description='XTask MT on NYU Dataset')
    parser.add_argument('--input_path', default='./data/nyu_preprocessed',
                        help='path of dataset (default: ./data/nyu_preprocessed)')
    parser.add_argument('--height', type=int, default=288,
                        help='height of output (default: 288)')
    parser.add_argument('--width', type=int, default=384,
                        help='width of output (default: 384)')

    parser.add_argument('-e', '--epochs', type=int, default=90,
                        help='number of epochs (default: 90)')
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
    parser.add_argument('--scheduler_step_size', type=int, default=60,
                        help='step size of scheduler (steplr)')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5,
                        help='lr decay of scheduler (steplr)')

    parser.add_argument('-a', '--alpha', type=float, default=0.0001,
                        help='alpha of loss function (default: 0.0001)')
    parser.add_argument('-g', '--gamma', type=float, default=0.0001,
                        help='gamma of loss function (default: 0.0001')
    parser.add_argument('--label_smoothing', type=float, default=0.,
                        help='label smoothing when calculating KL loss')
    parser.add_argument('--lp', default="L1", choices=["MSE", "L1", "logL1"],
                        help='depth loss for depth loss')
    parser.add_argument('--tseg_loss', default="cross", choices=["cross", "kl"],
                        help='label loss for cross-task segmt loss')
    parser.add_argument('--tdep_loss', default='L1', choices=["ssim", "L1"])

    parser.add_argument('--is_shallow', action='store_true',
                        help='flag: use shallow decoder for each task')
    parser.add_argument('--batch_norm', action='store_true',
                        help='flag: enable batch normalization in ttnet')

    parser.add_argument('--uncertainty_weights', action='store_true',
                        help='flag: use uncertainty weights (Kendall+, 2018) for balancing cross-task losses')
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

def cityscapes_xtask_parser():

    parser = argparse.ArgumentParser(description='XTask MT on Cityscapes Dataset')
    parser.add_argument('--input_path', default='./data/cityscapes',
                        help='path of dataset (default: ./data/cityscapes)')
    parser.add_argument('--height', type=int, default=128,
                        help='height of output (default: 128)')
    parser.add_argument('--width', type=int, default=256,
                        help='width of output (default: 256)')

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
    parser.add_argument('-n', '--num_classes', type=int, default=7, choices=[7, 19],
                        help='number of classes for segmentation task (default: 7)')
    parser.add_argument('--scheduler_step_size', type=int, default=60,
                        help='step size of scheduler (steplr)')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5,
                        help='lr decay of scheduler (steplr)')

    parser.add_argument('-a', '--alpha', type=float, default=0.01,
                        help='alpha of loss function (default: 0.01)')
    parser.add_argument('-g', '--gamma', type=float, default=0.01,
                        help='gamma of loss function (default: 0.01')
    parser.add_argument('-t', '--temp', type=int, default=1,
                        help='temperature of distillation (default: 1)')
    parser.add_argument('--label_smoothing', type=float, default=0.,
                        help='label smoothing when calculating KL loss')
    parser.add_argument('--lp', default="L1", choices=["MSE", "L1", "logL1", "smoothL1"],
                        help='depth loss for depth loss')
    parser.add_argument('--tdep_loss', default='L1', choices=["ssim", "L1"])
    parser.add_argument('--tseg_loss', default="cross", choices=["cross", "kl"],
                        help='label loss for cross-task segmt loss')

    parser.add_argument('--is_shallow', action='store_true',
                        help='flag: use shallow decoder for each task')
    parser.add_argument('--batch_norm', action='store_true',
                        help='flag: enable batch normalization in ttnet')

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