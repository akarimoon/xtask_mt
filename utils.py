import os
from datetime import datetime

def write_results(logger, args, model, file_path="./tmp/output/results.txt", exp_num=None):
    with open(file_path, 'a') as f:
        f.write("=" * 10 + "\n")
        if exp_num is not None:
            f.write("Experiment #{}\n".format(exp_num))
        f.write("Parameters: enc={}, lr={}, beta={}, lp={}, tsegmt={}, alpha={}, gamma={}, smoothing={}\n".format(
            args.enc_layers, args.lr, (args.b1, args.b2), args.lp, args.tseg_loss, args.alpha, args.gamma, args.label_smoothing
        ))
        if args.num_classes != 19:
            f.write("# of classes: {}\n".format(args.num_classes))
        f.write("transfernet type: {}, use_uncertainty: {}, use gradloss: {}\n".format(
            model.trans_name, args.uncertainty_weights, args.grad_loss))
        print_segmt_str = "Pix Acc: {:.3f}, Mean acc: {:.3f}, IoU: {:.3f}\n"
        f.write(print_segmt_str.format(
            logger.glob, logger.mean, logger.iou
        ))

        print_depth_str = "Scores - RMSE: {:.4f}, iRMSE: {:.4f}, Abs Rel: {:.4f}, Sqrt Rel: {:.4f}, " +\
            "delta1: {:.4f}, delta2: {:.4f}, delta3: {:.4f}\n"
        f.write(print_depth_str.format(
            logger.rmse, logger.irmse, logger.abs_rel, logger.sqrt_rel, logger.delta1, logger.delta2, logger.delta3
        ))

def write_indv_results(args, model, folder_path):
    with open(os.path.join(folder_path, "indv_results.txt"), "a") as f:
        now = datetime.now()
        f.write("Date: {}\n".format(now.strftime("%b-%d-%Y %H:%M:%S")))
        f.write("arguments:\n")
        f.write("   predicting at size [{}*{}]\n".format(args.height, args.width))
        f.write("   batch size: {}, train for {} epochs\n".format(args.batch_size, args.epochs))
        f.write("   enc={}, numclasses={}, lr={}, beta={}, lp={}, tsegmt={}, alpha={}, gamma={}, smoothing={}\n".format(
            args.enc_layers, args.num_classes, args.lr, (args.b1, args.b2), args.lp, args.tseg_loss,
            args.alpha, args.gamma, args.label_smoothing
        ))
        f.write("   transfernet type: {}, use_uncertainty: {}, use gradloss: {}\n".format(
            model.trans_name, args.uncertainty_weights, args.grad_loss))

def make_results_dir(folder_path="./tmp"):
    i = 1
    while True:
        num = str(i).zfill(3)
        if not os.path.exists(os.path.join(folder_path, num)):
            os.mkdir(os.path.join(folder_path, num))
            break
        else:
            i += 1

    os.mkdir(os.path.join(folder_path, num, "model"))
    os.mkdir(os.path.join(folder_path, num, "output"))

    return num, os.path.join(folder_path, num)

