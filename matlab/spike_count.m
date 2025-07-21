clear all;
spk_nii_path = '/disks/Programming/simtb_ds/02_EXPERIMENT/spk_niis/';

files = dir(strcat(spk_nii_path , '*.nii'));
type(files.name)
out = cell(numel(files.name),2);

i = 1;
for file = files'
    nii = niftiread(strcat(spk_nii_path,file.name));
    
    out(i,1) = file.name;
    out(i,2) = sum(nii,'all');
    i = i + 1;
end
