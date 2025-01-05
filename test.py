import os
import argparse
from tqdm import tqdm
import torch
from torch import nn

from CO_Sal_dataLoader import Test_get_loader
from models.GCoNet_plus_vgg_pvt import CoSODnet
from util import save_tensor_img
from config import Config

def main(args):
    # Init model nn.DataParallel(CoSODnet()).cuda()
    config = Config()

    device = torch.device("cuda")
    model = nn.DataParallel(CoSODnet()).to(device)
    model.eval()
    # print('Testing with model {}'.format(args.ckpt))
    gconet_dict = torch.load(args.ckpt)
    model.load_state_dict(gconet_dict)



    for testset in args.testsets.split('+'):
        print('Testing {}...'.format(testset))
        root_dir = '../../Co-SOD dataset/'
        if testset == 'CoSal1k':
            test_img_path = root_dir + testset, '/RGB/'
            test_depth_path = root_dir + testset, '/D/'
            test_gt_path = root_dir + testset + '/GT/'
            saved_root = os.path.join(args.pred_dir, testset)
        elif testset == 'CoSal150':
            test_img_path = root_dir + testset, '/RGB/'
            test_depth_path = root_dir + testset, '/D/'
            test_gt_path = root_dir + testset + '/GT/'
            saved_root = os.path.join(args.pred_dir, testset)
        elif testset == 'CoSeg183':
            test_img_path = root_dir + testset, '/RGB/'
            test_depth_path = root_dir + testset, '/D/'
            test_gt_path = root_dir + testset + '/GT/'
            saved_root = os.path.join(args.pred_dir, testset)
        else:
            print('Unkonwn test dataset')
            print(args.dataset)
        
        test_loader = get_loader(
            test_img_path, test_depth_path, test_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True)

        for batch in tqdm(test_loader):
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]
            with torch.no_grad():
                scaled_preds = model(inputs)[-1]

            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            num = len(scaled_preds)
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                if config.db_output_refiner or (not config.refine and config.db_output_decoder):
                    res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear', align_corners=True)
                else:
                    res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear', align_corners=True).sigmoid()
                save_tensor_img(res, os.path.join(saved_root, subpath))


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model',default='SgCoSOD',type=str,help="Options: '', ''")

    parser.add_argument('--testsets',default='CoCA+CoSOD3k+CoSal2015',type=str,
                        help="Options: 'CoCA','CoSal2015','CoSOD3k','iCoseg','MSRC'")

    parser.add_argument('--size',default=288,type=int,help='input size')

    parser.add_argument('--ckpt', default='./ckpt/Best_Sm_epoch.pth', type=str, help='model folder')
    parser.add_argument('--pred_dir', default='./preds/', type=str, help='Output folder')

    args = parser.parse_args()

    main(args)
