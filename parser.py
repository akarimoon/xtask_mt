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


def cityscapes_parser():

    parser = argparse.ArgumentParser(description='Hierarchical MT on Cityscapes Dataset')
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
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--b1', '--beta_1', type=float, default=0.9,
                        help='beta_1 of Adam (default: 0.9)')
    parser.add_argument('--b2', '--beta_2', type=float, default=0.99,
                        help='beta_2 of Adam (default: 0.99)')
    parser.add_argument('-a', '--alpha', type=float, default=0.4,
                        help='alpha of loss function (default: 0.4)')
    parser.add_argument('--save_weights', default='./tmp/model/hi_model.pth',
                        help='path to where weights are saved (default: ./tmp/model/hi_model.pth)')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--infer_only', action='store_true', help='flag: only infer')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--debug', action='store_true', help='flag: debug mode, dont save weights')
    args = parser.parse_args()

    return args

def cityscapes_inference_parser():
    parser = argparse.ArgumentParser(description='Inference on an image')
    parser.add_argument(
        '--im', dest='im_file', help='input image', default=None, type=str
    )
    parser.add_argument(
        '--rpn-pkl', dest='rpn_pkl', help='rpn model file (pkl)',
        default="https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl",
        type=str
    )
    parser.add_argument(
        '--rpn-cfg', dest='rpn_cfg', help='cfg model file (yaml)',
        default="../detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml", type=str
    )
    parser.add_argument(
        '--output-dir', dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer)', default='/tmp/infer',
        type=str
    )
    parser.add_argument(
        'models_to_run',
        help='pairs of models & configs, listed like so: [pkl1] [yaml1] [pkl2] [yaml2] ...',
        default=None, nargs=argparse.REMAINDER
    )
    args = parser.parse_args()

    return args
