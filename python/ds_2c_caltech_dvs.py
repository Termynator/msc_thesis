import os
import sys
import glob

import numpy as np
from sklearn.model_selection import train_test_split

path = '/disks/Programming/msc_thesis-master/'
data_path = 'dvs_data/Caltech101/'
npy_path = 'numpys/'

np.random.seed(42)

trn_size = 0.8
tst_size = 0.2

all_files = np.empty([0,],dtype=object)

for root,dirs,files in os.walk(os.path.join(path,data_path)):
  files = np.asarray(files,dtype=object)
  for i,_ in enumerate(files):
    files[i] = os.path.relpath(os.path.join(root,files[i]), start=path)
    print(f"Saving relative path: {files[i]}")
  all_files = np.append(all_files,files,axis = 0)

sel_classes = ['airplanes_100','Motorbikes_100'] # roughly 100 images per class
all_labls =  np.empty_like(all_files)

for i, file_path in enumerate(all_files):
  label = os.path.basename(os.path.dirname(file_path))
  all_labls[i] = label

sel_files = np.empty([0,])
sel_labls = np.empty([0,])
for cls in sel_classes:
  idx =  np.where(all_labls == cls)
  sel_files = np.append(sel_files,all_files[idx])
  sel_labls = np.append(sel_labls,all_labls[idx])

trn_files,tst_files,trn_labls,tst_labls = train_test_split(sel_files, sel_labls, train_size=trn_size, random_state=42, stratify = sel_labls)

trn = np.stack([trn_files,trn_labls])
tst = np.stack([tst_files,tst_labls])

print(trn.shape)
print(tst.shape)



np.save(os.path.join(path,npy_path,'trn_2c_cal_dvs.npy'), trn)
np.save(os.path.join(path,npy_path,'tst_2c_cal_dvs.npy'), tst)
