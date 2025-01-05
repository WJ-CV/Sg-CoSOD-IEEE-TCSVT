import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default='GCoNet_plus', type=str, help="Options: '', ''")
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--trainset', default='DUTS_class', type=str, help="Options: 'DUTS_class'")
parser.add_argument('--size', default=288, type=int, help='input size')
parser.add_argument('--group_size', default=10, type=int, help='group size')
parser.add_argument('--group_num', default=2, type=int, help='group size')
parser.add_argument('--decay_step_size', default=70, type=int, help='decay_step_size')
parser.add_argument('--ckpt_dir', default='./ckpt/', help='Temporary folder')

parser.add_argument('--train_root_dir', default='', help='')
parser.add_argument('--test_root_dir', default='', help='')

parser.add_argument('--testsets', default=['CoSOD3k','CoSal2015', 'CoCA'], type=str, help="Options: , 'CoSal150', 'CoSal1k'") #'CoSal1k','CoSal150', 'CoSeg183'
parser.add_argument('--val_dir', default='./val_result/', type=str, help="Dir for saving tmp results for validation.")
parser.add_argument('--test_dir', default='./test_result/', type=str, help="Dir for saving results for test.")
parser.add_argument('--gpu_id', type=str, default='2', help='train use gpu')
args = parser.parse_args()