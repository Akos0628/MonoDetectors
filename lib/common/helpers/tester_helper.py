import os
import tqdm

import torch
import numpy as np

import shutil
from datetime import datetime



from lib.common.helpers.eval_helper import eval_from_scrach
from lib.monoLSS.helpers.save_helper import load_checkpoint
class Tester(object):
    def __init__(self, cfg_tester, cfg_dataset, model, split, logger):
        self.cfg = cfg_tester
        self.logger = logger
        self.label_dir = cfg_dataset['label_dir']
        self.eval_cls = cfg_dataset['eval_cls']

        # data split loading
        assert split in ['train', 'val', 'trainval', 'test']
        self.split = split
        split_dir = os.path.join(cfg_dataset['root_dir'], cfg_dataset['data_dir'], 'ImageSets', split + '.txt')
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]


    def test(self, printHelper, printer):
        results = {}
        progress_bar = tqdm.tqdm(total=len(self.idx_list), leave=True, desc='Evaluation Progress')
        for idx in self.idx_list:
            img, calibs = printHelper.getPrintables(idx)
            preds = printer.print(img, calibs, 0.2)

            
            results.update(preds)
            progress_bar.update()

        output_dir = os.path.join(
            self.cfg['out_dir'],
            os.path.basename(os.path.splitext(self.cfg['resume_model'])[0])
        )
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        self.save_results(results, output_dir=output_dir)
        progress_bar.close()

        eval_from_scrach(
            self.label_dir,
            os.path.join(output_dir, 'data'),
            self.eval_cls,
            ap_mode=40
        )


    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)
        for img_id in results.keys():
            out_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            f = open(out_path, 'w')
            for i in range(len(results[img_id])):
                #class_name = self.class_name[int(results[img_id][i][0])]
                #f.write('{} 0.0 0'.format(class_name))
                for j in range(0, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()







