import os
import sys
import glob

import numpy as np
from sklearn.model_selection import train_test_split

path = '/home/zeke/Programming/msc_thesis-master/'
data_path = 'simtb_data/diff_comp/niis/'
big_disk_data_path = '/disks/Programming/simtb_ds/diff_comp/niis/'
npy_path = 'numpys/'

np.random.seed(42)

trn_size = 0.8
tst_size = 0.2

#files = os.listdir(os.path.join(path,data_path))
files = os.listdir(big_disk_data_path)
labls = np.empty_like(files)

for i in range(len(labls)):
 labls[i] = files[i][0]
 #files[i] = os.path.join(path,data_path,files[i])
 files[i] = os.path.join(big_disk_data_path,files[i])


trn_files,tst_files,trn_labls,tst_labls = train_test_split(files, labls, train_size=trn_size, random_state=42)

trn = np.stack([trn_files,trn_labls])
tst = np.stack([tst_files,tst_labls])

print(trn.shape)
print(tst.shape)

np.save(os.path.join(path,npy_path,'trn_simtb_dc.npy'), trn)
np.save(os.path.join(path,npy_path,'tst_simtb_dc.npy'), tst)
