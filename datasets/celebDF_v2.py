#-*- coding: utf-8 -*-
import os

from glob import glob
import numpy as np

from .builder import DATASETS
from .common import CommonDataset


@DATASETS.register_module()
class CDFV2(CommonDataset):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def _load_from_path(self, split):
        assert os.path.exists(self._cfg.DATA[self.split.upper()].ROOT), "Root path to dataset can not be None!"
        data = self._cfg["DATA"]
        data_type = data.TYPE
        fake_types = self._cfg.DATA[split.upper()]["FAKETYPE"]
        img_paths, labels, mask_paths, ot_props = [], [], [], []
        
        count = 0
        n_samples = 100000

        # Load image data for each type of fake techniques
        for idx, ft in enumerate(fake_types):
            data_dir = os.path.join(self._cfg.DATA[self.split.upper()].ROOT, self.split, data_type, ft)
            if not os.path.exists(data_dir):
                raise ValueError("Data Directory can not be invalid!")
            
            for sub_dir in os.listdir(data_dir):
                sub_dir_path = os.path.join(data_dir, sub_dir)
                img_paths_ = glob(f'{sub_dir_path}/*.{self._cfg.IMAGE_SUFFIX}')
                
                if 'Celeb-synthesis' in ft:
                    if count < n_samples:
                        n_add = len(img_paths_) if ((n_samples-count) > len(img_paths_)) else (n_samples-count)
                        count += n_add
                        print(f'n fake samples added --- {count}')
                    else:
                        continue
                else:
                    n_add = len(img_paths_)

                img_paths.extend(img_paths_[:n_add])
                labels.extend(np.full(n_add, int(ft == 'Celeb-synthesis')))
                
        print('{} image paths have been loaded from CDFv2!'.format(len(img_paths)))          
        return img_paths, labels, mask_paths, ot_props
