%% SMALL DIFFERENCE DC
% quantify % difference between the two classes?
% 100x100

clear all;
tic

% Paths
addpath(genpath('simtb/'))
mat_path = '/disks/Programming/simtb_ds/01_EXPERIMENT/mats/';
spk_nii_path = '/disks/Programming/simtb_ds/01_EXPERIMENT/spk_niis/';
ica_nii_path = '/disks/Programming/simtb_ds/01_EXPERIMENT/ica_niis/';



% Param files and make mats
sP = simtb_create_sP('comp_A_params');
sP.out_path = mat_path;
simtb_main(sP);

sP = simtb_create_sP('comp_B_params');
sP.out_path = mat_path;
simtb_main(sP);

%make ica niis
sP = simtb_create_sP('comp_A_params');
sP.out_path = ica_nii_path;
sP.saveNII_FLAG = 1;
simtb_main(sP);

sP = simtb_create_sP('comp_B_params');
sP.out_path = ica_nii_path;
sP.saveNII_FLAG = 1;
simtb_main(sP);


timeStepS = 0.001;                  % 1 msec
                    
durationS = 0.050; % 0.100              % simulation duration
times = 0:timeStepS:durationS;	% a vector with each time step		

files = dir(strcat(mat_path , '*DATA.mat'));

for file = files'
    load(strcat(mat_path,file.name));
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
    niftiwrite(nii,strcat(spk_nii_path,erase(file.name,'_DATA.mat'),'.nii'));
end
toc

