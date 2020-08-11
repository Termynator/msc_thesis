clear all;
addpath(genpath('simtb/'))
%make params
tic
sP = simtb_create_sP('exp_diff_comp_A_params');
simtb_main(sP);

sP = simtb_create_sP('exp_diff_comp_B_params');
simtb_main(sP);
toc
