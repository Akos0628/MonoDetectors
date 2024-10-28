import os
import tqdm

import shutil


from lib.common.helpers.eval_helper import eval_from_scrach
class Tester(object):
    def __init__(self, cfg_tester, cfg_dataset, split, logger):
        self.cfg = cfg_tester
        self.logger = logger
        self.label_dir = cfg_dataset['label_dir']
        self.eval_cls = cfg_dataset['eval_cls']

        # data split loading
        assert split in ['train', 'val', 'trainval', 'test']
        self.split = split
        split_dir = os.path.join(
            cfg_dataset['root_dir'], 
            cfg_dataset['data_dir'],
            'testing' if split == 'test' else 'training',
            'ImageSets', split + '.txt'
        )
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]


    def test(self, printHelper, printer):
        output_dir = self.cfg['out_dir']
        if os.path.exists(output_dir):
            print("delete output")
            shutil.rmtree(output_dir)
        
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)

        progress_bar = tqdm.tqdm(total=len(self.idx_list), leave=True, desc='Evaluation Progress')
        for idx in self.idx_list:
            idx = int(idx)
            img, calibs = printHelper.getPrintables(idx)
            preds = printer.print(img, calibs)
            
            self.save_result(idx, preds, output_dir=output_dir)
            progress_bar.update()

        
        progress_bar.close()

        eval_from_scrach(
            self.label_dir,
            output_dir,
            self.eval_cls,
            ap_mode=40
        )


    def save_result(self, idx, results, output_dir='./outputs'):
        out_path = os.path.join(output_dir, '{:06d}.txt'.format(idx))
        f = open(out_path, 'w')
        for i in range(len(results)):
            class_name = results[i][0]
            f.write('{} 0.0 0'.format(class_name))
            for j in range(3, len(results[i])):
                f.write(' {:.2f}'.format(results[i][j]))
            f.write('\n')
        f.close()







