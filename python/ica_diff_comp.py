import os
import glob

from nilearn.decomposition import CanICA

data_path = '/disks/Programming/simtb_ds/diff_comp/ica_niis/'

niis = glob.glob(os.path.join(data_path,'*.nii'))

canica = CanICA(n_components=20,
                memory="nilearn_cache", memory_level=2,
                verbose=10,
                mask_strategy='background',
                random_state=0)
canica.fit(niis)
