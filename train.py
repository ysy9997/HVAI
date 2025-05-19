import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--configs', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--gpu_id', type=str, default=None)
    parser.add_argument('--model', type=str, default='')
    args = parser.parse_args()