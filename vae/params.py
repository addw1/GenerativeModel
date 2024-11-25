import argparse

def generate_parser():
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of VAE for MNIST")
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--z-dim', type=int, default=2,
                        help='dimension of hidden variable Z (default: 2)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='interval between logs about training status (default: 100)')
    parser.add_argument('--learning-rate', type=int, default=1e-3,
                        help='learning rate for Adam optimizer (default: 1e-3)')
    parser.add_argument('--prr', type=bool, default=True,
                        help='Boolean for plot-reproduce-result (default: True')
    parser.add_argument('--prr-z1-range', type=int, default=2,
                        help='z1 range for plot-reproduce-result (default: 2)')
    parser.add_argument('--prr-z2-range', type=int, default=2,
                        help='z2 range for plot-reproduce-result (default: 2)')
    parser.add_argument('--prr-z1-interval', type=int, default=0.2,
                        help='interval of z1 for plot-reproduce-result (default: 0.2)')
    parser.add_argument('--prr-z2-interval', type=int, default=0.2,
                        help='interval of z2 for plot-reproduce-result (default: 0.2)')
    return parser