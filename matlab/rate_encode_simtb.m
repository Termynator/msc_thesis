clear all;
addpath(genpath('simtb/'))

%load_path = '/home/zeke/Programming/msc_thesis/simtb_data/diff_comp/mats/';
%save_path = '/home/zeke/Programming/msc_thesis/simtb_data/diff_comp/niis/';
load_path = '/disks/Programming/simtb_ds/diff_comp/mats/';
save_path = '/disks/Programming/simtb_ds/diff_comp/niis/';


timeStepS = 0.001;                  % 1 msec
                    
durationS = 0.050; % 0.100              % simulation duration
times = 0:timeStepS:durationS;	% a vector with each time step		

files = dir(strcat(load_path , '*DATA.mat'));

tic
for file = files'
    file.name
    load(strcat(load_path,file.name));
    nii = zeros([1,size(D,2),size(D,3)]);
    for i=1:size(D,1)
        slice = squeeze(ceil(D(i,:,:)));
        slice_3d = zeros([size(times,2),size(slice,1),size(slice,2)]);
        for j = 1:size(times,2)
            slice_3d(j,:,:) = slice;
        end
        slice = slice_3d;
        vt = rand(size(slice));
        spikes = (slice.*timeStepS) > vt;
        
        nii = cat(1,nii,spikes);
    end
    %nii = nii[2:,:,:] remove zeros?
    nii = permute(nii,[2,3,1]);
    %size(nii)
    niftiwrite(nii,strcat(save_path,erase(file.name,'_DATA.mat'),'.nii'));
end
toc
