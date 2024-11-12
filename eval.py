import os
import sys
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import logging
import argparse

from lib.common.helpers.dataloader_helper import build_dataset
from lib.common.helpers.tester_helper import Tester
from lib.common.helpers.print_selector import selectPrinter
from lib.common.helpers.print_helper import PrintHelper


parser = argparse.ArgumentParser(description='implementation of MonoLSS')
parser.add_argument('--config', type=str, default='configs/kitti-multi.yaml')
args = parser.parse_args()


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def main():   
    assert (torch.cuda.is_available())

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # load cfg
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    os.makedirs(cfg['tester']['log_dir'], exist_ok=True)
    logger = create_logger(os.path.join(cfg['tester']['log_dir'], 'train.log'))

    #  build dataloader
    dataset = build_dataset(cfg['dataset'], "train")                           
        
    # build model
    printHelper = PrintHelper(dataset)
    printer = selectPrinter(cfg)

    # evaluation mode
    print('evaluation start')
    tester = Tester(cfg['tester'], cfg['dataset'], "val", logger)
    tester.test(printHelper, printer)
    return
    

if __name__ == '__main__':
    main()
