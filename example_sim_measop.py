import torch
import yaml
import os
import argparse
import numpy as np

from pysrc.io import read_uv
from pysrc.operator_tkbn import operator_tkbn
from pysrc.operator_pynufft import operator_pynufft

def parse_args():
    parser = argparse.ArgumentParser(description='Read uv and imweight files')
    parser.add_argument('--yaml_file', type=str, default='./configs/measop.yaml',
                    help='Path to the yaml file containing the arguments')
    return parser.parse_args()

def gen_measop(args):
    with open(args.yaml_file, 'r') as file:
        args.__dict__.update(yaml.load(file, Loader=yaml.FullLoader))
    if not hasattr(args, 'im_size'):
        args.im_size = (args.im_size_x, args.im_size_y)
        args.dict_save_path = None
    args.device = torch.device('cuda') if args.on_gpu and torch.cuda.is_available() else torch.device('cpu')
    # read Fourier sampling pattern from specified data file 
    uv, nWimag, data_dict = read_uv(args.data_file, args.superresolution, args.dict_save_path, args.device, args.nufft)
    # create measurement operator object based on the chosen nufft library
    match args.nufft:
        case 'tkbn':
            measop = operator_tkbn(im_size=args.im_size, op_type='table', op_acc='exact', device=args.device)
        case 'pynufft':
            measop = operator_pynufft(im_size=args.im_size, device=args.device)
        
    # set the Fourier sampling pattern in the measurement operator
    measop.set_uv_imweight(uv, None)
    return measop, nWimag, data_dict

if __name__ == '__main__':
    args = parse_args()
    measop, _, _ = gen_measop(args)