clear all;
%make params
sP = simtb_create_sP('experiment_params_aod');
sP.saveNII_FLAG = 0;
sP.verbose_display = 1;
%simulate
simtb_main(sP);
%show
load('/home/zeke/Programming/msc_thesis/matlab/simtb/simulations/aod_subject_001_DATA.mat')
load('/home/zeke/Programming/msc_thesis/matlab/simtb/simulations/aod_subject_001_SIM.mat')

simtb_movie(D, sP ,0,'none',[]);

