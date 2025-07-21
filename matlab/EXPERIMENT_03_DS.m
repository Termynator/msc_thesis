%% Different Duration Block Design
% quantify % difference between the two classes?


clear all;
tic

% Paths
addpath(genpath('simtb/'))
mat_path =     '/disks/Programming/simtb_ds/experiment_03/mats/';
spk_nii_path = '/disks/Programming/simtb_ds/experiment_03/spk_niis/';
ica_nii_path = '/disks/Programming/simtb_ds/experiment_03/ica_niis/';

spk_resolution = 80;
ica_resolution = 150;

% Param files and make mats
sP = simtb_create_sP('time_A_params_03');
sP.out_path = mat_path;
sP.saveNII_FLAG = 0;
sP.nV = spk_resolution;
simtb_main(sP);

sP = simtb_create_sP('time_B_params_03');
sP.out_path = mat_path;
sP.saveNII_FLAG = 0;
sP.nV = spk_resolution;
simtb_main(sP);

%make ica niis
sP = simtb_create_sP('time_A_params_03');
sP.out_path = ica_nii_path;
sP.saveNII_FLAG = 1;
sP.nV = ica_resolution;
simtb_main(sP);

sP = simtb_create_sP('time_B_params_03 ');
sP.out_path = ica_nii_path;
sP.saveNII_FLAG = 1;
sP.nV = ica_resolution;
simtb_main(sP);


timeStepS = 0.001;                  % 1 msec
                    
durationS = 0.050; % 0.100              % simulation duration
times = 0:timeStepS:durationS;	% a vector with each time step		

files = dir(strcat(mat_path , '*DATA.mat'));

for file = files'
    file.name
    load(strcat(mat_path,file.name));
    nii = zeros([1,size(D,2),size(D,3)]);
    size(D)
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
    niftiwrite(nii,strcat(spk_nii_path,erase(file.name,'_DATA.mat'),'.nii'));
end
toc
