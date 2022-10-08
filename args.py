import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str2bool, default=True)
    parser.add_argument("--dataset", type=str.upper, default="SYNTHETIC")
    parser.add_argument("--group", type=str, default="2-1", help="Required for SMD dataset.")
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=640)
    parser.add_argument("-learning_rate", type=float, default=0.0001)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True)
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--use_tensorboard", type=str2bool, default=True)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--clip", type=float, default=5)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.95,help="Schedular step rate")
    parser.add_argument("--theta", type=float, default=0.16,help="KLD weights")
    parser.add_argument("--n_latent", type=int, default=16)
    return parser