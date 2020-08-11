import os
import sys
import glob

import numpy as np
import data

root_path = '/home/zeke/Programming/msc_thesis-master/'
data_path = 'dvs_data/Caltech101/airplanes/'
path = os.path.join(root_path,data_path)

fnames = glob.glob(os.path.join(path,'*.bin'))

path = os.path.join(root_path,data_path)

for fname in fnames:
  print('Anal: ',fname[-24:])
  saccade = data.read_dataset(fname)
  saccade.show_em(frame_length = 500)
