% Generates a test fMRI data set and compares the performance of a variety of node discovery algorithms
% Compares results of PROFUMO and ICA
% Main aim is to test effects of spatial overlap vs spatial misalignment

% SPDX-License-Identifier: Apache-2.0

close all; clear all; clc

% Inputs
baseFileName = 'Results/Simulations';
saveData = true

% Key settings
overlap = true
misalignment = true
temporalCorrelations = true
structuredNoise = true

% Fix filename
if ~overlap
    baseFileName = strcat(baseFileName, '_NoOverlap');
end
if ~misalignment
    baseFileName = strcat(baseFileName, '_NoMisalignment');
end
if ~temporalCorrelations
    baseFileName = strcat(baseFileName, '_NoTemporalCorrelations');
end
if ~structuredNoise
    baseFileName = strcat(baseFileName, '_NoStructuredNoise');
end
baseFileName

% Plotting
plotFigures = false;

%% Set paths etc %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

restoredefaultpath();
initialiseEnvironment();
prettyFigures();

rng('shuffle');

%% Config initialisation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

params = struct();
options = struct();

% Get current commit
[status, params.git_commit] = system('git rev-parse HEAD');
if status ~= 0
    params.git_commit = '';
end
clear status

%% Set size of problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Number of times to repeat simulation / test cycle
params.nRepeats = 1; %10

%Details of scans - two (fast TR) 5 min runs per subject
params.S = 2; %40       %Subjects
params.R = 2*ones(params.S,1);   %Runs

params.T  = 500;     %No. of time points per fMRI scan
params.TR = 2.0;
params.dt = 0.1;     %Neural sampling rate
%Amount of neural points to simulate - more than scan length so don't have
%to zero pad HRF at start of scan
params.Tn = ceil(1.25 * params.T * params.TR / params.dt);

%% Set size of atlas / mode matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Atlas
atlasParams = params;
atlasParams.V = 10000;    %Voxels
atlasParams.N = 100;      %Number of nodes in the atlas

% Modes
modeParams = params;
modeParams.V = atlasParams.N;
modeParams.N = 15;        %Number of modes

% And store appropriate values for combined maps
params.N = modeParams.N;
params.V = atlasParams.V;

%Number of modes to infer
params.iN = 18;

%% Set the details of the tests we want to do %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
%Atlas

atlasOptions = struct();
atlasOptions.P.form = 'Probabilistic';

%Choose form for Pg
atlasOptions.Pg.form = 'BlockAtlas';
% widthPrecision controls the variability in the size of the parcels
% Smaller numbers give more variability in parcel sizes
atlasOptions.Pg.widthPrecision = 25;
% Post-hoc smoothing of maps (width, in voxels, of the filter)
atlasOptions.Pg.smootherWidth = 0.1 * (atlasParams.V / atlasParams.N);

%Choose form for Ps
atlasOptions.Ps.form = 'WeightedGamma';
% Probability subject voxel is not drawn from group distribution
if overlap
    atlasOptions.Ps.p = 0.0005;
else
    atlasOptions.Ps.p = 0.0;
end
% Minimum weight - useful to make sure all weights are different from noise
atlasOptions.Ps.minWeight = 0.1;
% Weights are gamma(a,b) distributed (mean = a/b)
% Increasing a,b makes them more stable
atlasOptions.Ps.weightRange.a = 0.9 * 20.0;
atlasOptions.Ps.weightRange.b = 20.0;
atlasOptions.Ps.epsilon = 0.0;

% Choose registration
if misalignment
    atlasOptions.P.registration.form = 'RandomSmooth';
    atlasOptions.P.registration.maxError = 1.5 * (atlasParams.V / atlasParams.N);
    % This parameter controls the size of misalignments
    % It represents the furthest one voxel can be moved by misregistration
    % Useful to express this in terms of `c * (atlas.V / atlas.N)`, i.e. the
    % average parcel size. The parameter `c` then represents the max
    % misalignment in terms of number of parcels rather than voxels.
else
    atlasOptions.P.registration.form = 'Null';
end

%--------------------------------------------------------------------------
%Modes

modeOptions = struct();
modeOptions.P.form = 'Probabilistic';

modeOptions.Pg.form = 'BiasedBoxcar';
% How many spatially contiguous blocks per mode? Follows `Poisson(nBlocks) + 1`
modeOptions.Pg.nBlocks = 0.5;
% How big are the modes? On average, they cover `p * V` voxels
% If we have N modes, then we expect `p * N` modes in every voxel
% This is therefore a crude proxy for overlap
%%% HighOverlap: 1.4; LowOverlap 1.2; %%%
if overlap
    modeOptions.Pg.p = 1.4 / params.N;
else
    modeOptions.Pg.p = 0.95 / params.N;
end
modeOptions.Pg.pVar = 0.01 ^ 2; % i.e. p will vary over approximately +/- 2.0 * sqrt(pVar)
% Proportion of (secondary) blocks that are positive
modeOptions.Pg.pPosBlock = 0.5;

%Choose form for Ps
modeOptions.Ps.form = 'WeightedGamma';
% Probability subject voxel is not drawn from group distribution
% `p = c / (V * N)` means that, on average `c` parcels are active
% in a given subject that were not in the group maps
if overlap
    modeOptions.Ps.p = 2.0 / (modeParams.V * modeParams.N);
else
    modeOptions.Ps.p = 0.0;
end
% Minimum weight - useful to make sure all weights are different from noise
modeOptions.Ps.minWeight = 0.0;
% Weights are gamma(a,b) distributed (mean = a/b)
% Increasing a,b makes them more stable
modeOptions.Ps.weightRange.a = 5.0;
modeOptions.Ps.weightRange.b = 5.0;
modeOptions.Ps.epsilon = 0.0;

%Choose registration errors
modeOptions.P.registration.form = 'Null';

%--------------------------------------------------------------------------
%Time courses

options.An.form = 'Freq';
% Amplitude variability
options.An.amp.a = 10.0;
options.An.amp.b = 10.0;
% Proportion of modes missing (per subject & mode)
options.An.offRate = 0.05 * 1/params.N;  % 1/20 subjects has a missing mode
switch options.An.form
    case 'Freq'
        % Reducing these parameters will increase the
        % strength of the correlations at the group,
        % subject and run level respectively
        if temporalCorrelations
            options.An.Cg_dof = 50;
            options.An.Cs_dof = 75;
            options.An.Cr_dof = 75;
        else
            options.An.Cg_dof = 1.0e5;
            options.An.Cs_dof = 1.0e5;
            options.An.Cr_dof = 1.0e5;
        end
        options.An.p = 0.1;
        options.An.fc = 0.1; %in Hz
        options.An.fAmp = 2;
        options.An.epsilon = 0.1;
end

%--------------------------------------------------------------------------
%BOLD signal

options.BS.form = 'SaturatingFlobsHRF';
% Little bit of Gaussian noise for old times sake
options.BS.SNR_P = 7.5;
options.BS.SNR_A = 7.5;
switch options.BS.form
    case 'Linear'
        
    case 'FlobsHRF'
        options.BS.HRFcoeffs.mu = [1 0 -0.2];
        options.BS.HRFcoeffs.sigma = [0.1 0.1 0.1];
        
    case 'SaturatingFlobsHRF'
        options.BS.HRFcoeffs.mu = [1 0 -0.2];
        options.BS.HRFcoeffs.sigma = [0.1 0.1 0.1];
        options.BS.tanhPercentile = 99;
        options.BS.tanhMax = 0.9;
end

%--------------------------------------------------------------------------
%Noise

%Signal to noise ratio (expressed in terms of power)
options.D.SNR = 0.1;
% stddev of spatial Gaussian smoothing kernel (in voxels)
% Has to be fairly big to account for the difference between 1D & 3D
options.D.sigma_s = 4.0;
% stddev of temporal Gaussian smoothing kernel (in voxels)
options.D.sigma_t = 0.75 / sqrt(params.TR);

%Type
if structuredNoise
    options.D.noise.form = 'StructuredTdist';
else
    options.D.noise.form = 'SpatiotemporallyWhiteTdist';
end
switch options.D.noise.form
    case 'SpatiotemporallyWhiteGaussian'
        
    case 'SpatiotemporallyWhiteTdist'
        options.D.noise.a = 3.0;
        options.D.noise.b = 3.0;
        
    case 'StructuredTdist'
        options.D.noise.a = 3.0;
        options.D.noise.b = 3.0;
        % Structured noise parameters
        options.D.noise.N = 5;
        % Variance of structured subspace relative to unstructured
        % Transformation expresses the variance of the signal v structured noise *per component*
        options.D.noise.structuredSNR = ...
            0.75 * options.D.SNR * (options.D.noise.N / params.N);
end

%--------------------------------------------------------------------------

%% Run the tests %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

scores = struct();
for n = 1:params.nRepeats
    
    repeat = sprintf('%02d', n)
    repeatDir = [baseFileName '_' repeat];
    mkdir(repeatDir);
    
    %% Generate data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if plotFigures && (n==1)
        plotNow = true;
    else
        plotNow = false;
    end
    
    atlasP = generateMaps(atlasParams, atlasOptions, plotNow);
    
    [modeP, plotMaps] = generateMaps(modeParams, modeOptions, plotNow);
    
    %Combine to make full maps
    P = cell(params.S,1);
    for s = 1:params.S
        P{s} = atlasP{s} * modeP{s};
    end
    %Plot if requested
    if plotNow
        plotMaps(P, params, options);
    end
    
    An = generateNeuralTimecourses(params, options, plotNow);
    
    [PA, A] = generateBoldSignal(P, An, params, options, plotNow);
    
    D = generateData(PA, params, options, plotNow);
    
    % Calculate netmats
    pcA  = calculateNetmats(A, params);
    
    %Finally, add a global rescaling such that all scans are overall
    %unit variance
    vD = 0;
    for s = 1:params.S
        for r = 1:params.R(s)
            vD = vD + var(D{s}{r}(:));
        end
    end
    vD = vD / sum(params.R);
    for s = 1:params.S
        for r = 1:params.R(s)
            D{s}{r} = D{s}{r} / sqrt(vD);
            PA{s}{r} = PA{s}{r} / sqrt(vD);
        end
    end
    
    %% Save simulated maps etc %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if saveData
        save(fullfile(repeatDir, 'SimParams.mat'), ...
            'P', 'A', 'An', 'pcA', ...
            'params', 'atlasParams', 'modeParams', ...
            'options', 'atlasOptions', 'modeOptions', ...
            '-v7.3');
            % D saved as NIfTIs
            %D = loadNIfTIs(niftiDir, params);
            % `Pg` etc can be reconstructed
    end
    
    %% Save NIFTIs for MELODIC and PROFUMO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    niftiDir = fullfile(repeatDir, 'NIfTIs');
    mkdir(niftiDir);
    saveNIfTIs(D, niftiDir, params);
    
    %% Accuracy from true group maps %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % No obvious way to combine group maps, so skip if fewer inferred
    if params.iN >= params.N
        try
            %Extract mean group map
            Pg = 0;
            for s = 1:params.S
                Pg = Pg + P{s};
            end
            Pg = Pg / params.S;
            % And save
            mapFile = fullfile(repeatDir, 'GT_Pg.nii.gz');
            save_avw( ...
                reshape(Pg, 10, 10, params.V / 100, params.N), ...
                mapFile, 'f', [1 1 1 1]);
            
            % Record actual size - don't include any extra maps asked for
            % This is basically equivalent to setting extras to zero, but more
            % numerically stable
            gt_params = params; gt_params.iN = gt_params.N;
            
            % Dual-reg from ground truth group maps
            dualregDir = fullfile(repeatDir, 'GT_Pg.dr');
            system(sprintf( ...
                'sh Methods/DualRegression.sh %s %s %s', ...
                dualregDir, niftiDir, mapFile));
            % `t` means thresholded as per Bijsterbosch et al., eLife, 2019
            [gtdr_P, gtdr_A, gtdr_Pt, gtdr_At] ...
                = loadDualRegression(dualregDir, gt_params);
            gtdr_pcA  = calculateNetmats(gtdr_A, gt_params);
            gtdr_pcAt = calculateNetmats(gtdr_At, gt_params);
            
            scores.GTg_DR(n) = calculateDecompositionAccuracy( ...
                    P, gtdr_P, A, gtdr_A, pcA, gtdr_pcA, gt_params);
            scores.GTg_DRt(n) = calculateDecompositionAccuracy( ...
                    P, gtdr_Pt, A, gtdr_At, pcA, gtdr_pcAt, gt_params);
            
            % Add dummy maps for any extras inferred
            %if params.iN > params.N
            %    Pg = [Pg, zeros([params.V, params.iN - params.N])];
            %end
            
            % Dual-reg from ground truth group maps
            %[gtdr_P, gtdr_A] = runDR(D, Pg, params);
            %gtdr_pcA = calculateNetmats(gtdr_A, gt_params);
            %scores.GTg_DR(n) = calculateDecompositionAccuracy( ...
            %        P, gtdr_P, A, gtdr_A, pcA, gtdr_pcA, gt_params);
            
            % Repeat, but now 'clean' dual regression (i.e. from PA, not D)
            %[gtdr_P, gtdr_A] = runDR(PA, Pg, params);
            %gtdr_pcA = calculateNetmats(gtdr_A, params);
            %scores.GTg_PADR(n) = calculateDecompositionAccuracy( ...
            %        P, gtdr_P, A, gtdr_A, pcA, gtdr_pcA, params);
            
            %Ground truth linear model
            %scores.GT.PA(n) = calculateBoldRecovery(PA, makePA(P,A,params), params);
            % N.B. This is essentially deprecated now there is noise in the PA subspace
        catch
            warning('Ground-truth dual regression failed.');
        end
        
    end
end
    
    %% Run MELODIC / DR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     
%     try
%         melodicDir = fullfile(repeatDir, 'MELODIC.gica');
%         system(sprintf( ...
%             'sh Methods/MELODIC.sh %s %s %d %1.2f', ...
%             melodicDir, niftiDir, params.iN, params.TR));
%         dualregDir = fullfile(repeatDir, 'MELODIC.dr');
%         system(sprintf( ...
%             'sh Methods/DualRegression.sh %s %s %s', ...
%             dualregDir, niftiDir, fullfile(melodicDir, 'melodic_IC.nii.gz')));
%         
%         % `t` means thresholded as per Bijsterbosch et al., eLife, 2019
%         icadr_Pg = loadMELODIC(melodicDir, params);
%         [icadr_P, icadr_A, icadr_Pt, icadr_At] ...
%             = loadDualRegression(dualregDir, params);
%         %[icadr_P, icadr_A] = runDR(D, icadr_Pg, params);
%         icadr_pcA  = calculateNetmats(icadr_A, params);
%         icadr_pcAt = calculateNetmats(icadr_At, params);
%         
%         scores.ICA_DR(n) = calculateDecompositionAccuracy( ...
%                 P, icadr_P, A, icadr_A, pcA, icadr_pcA, params);
%         scores.ICA_DRt(n) = calculateDecompositionAccuracy( ...
%                 P, icadr_Pt, A, icadr_At, pcA, icadr_pcAt, params);
%     catch
%         warning('MELODIC run failed.');
%     end
%     
%     %% Run PROFUMO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     
%     try
%         pfmDir = fullfile(repeatDir, 'PROFUMO.pfm');
%         system(sprintf( ...
%             'sh Methods/PROFUMO.sh %s %s %d %1.2f', ...
%             pfmDir, niftiDir, params.iN, params.TR));
%         
%         [pfm_Pg, pfm_P, pfm_A, pfm_pcA] = loadPROFUMO(pfmDir, params);
%         
%         scores.PROFUMO(n) = calculateDecompositionAccuracy( ...
%                 P, pfm_P, A, pfm_A, pcA, pfm_pcA, params);
%         % Correlations compared to 'neural' process
%         %pcAn = calculateNetmats(An, params);
%         % Feed `pcAn` instead of `pcA`
%         
%         
%         % Dual-reg from PFM group maps
%         %[pfmdr_P, pfmdr_A] = runDR(D, pfm_Pg, params);
%         %pfmdr_pcA = calculateNetmats(pfmdr_A, params);
%         
%         %scores.PFM_DR(n) = calculateDecompositionAccuracy( ...
%         %        P, pfmdr_P, A, pfmdr_A, pcA, pfmdr_pcA, params);
%     catch
%         warning('PROFUMO run failed.');
%     end
%     
%     %% Save scores %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     
%     save([baseFileName '.mat'], ...
%         'scores', ...
%         'params', 'atlasParams', 'modeParams', ...
%         'options', 'atlasOptions', 'modeOptions', ...
%         '-v7.3');
%     
%     %% And delete everything we don't need saved %%%%%%%%%%%%%%%%%%%%%%%%%%
%     if ~saveData
%         rmdir(repeatDir, 's');
%     end
%     
% end

%% Plot results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% try
%     scores = orderfields(scores, ...
%         {'PROFUMO', 'ICA_DR', 'ICA_DRt', 'GTg_DR', 'GTg_DRt'});
% catch
% end
% 
% plotScores(scores, params, baseFileName, plotFigures);
% if plotFigures
%     input('Press return to continue');
%     close all
% end
% 
% fprintf('Finished! Figures have been saved to:\n');
% fprintf('    %s/\n', baseFileName);
% fprintf('To replot the results you can now run e.g.:\n');
% fprintf('>> initialiseEnvironment();\n');
% fprintf('>> prettyFigures();\n');
% fprintf('>> load(''%s.mat'');\n', baseFileName);
% fprintf('>> plotScores(scores, params);\n');
% fprintf('\n');

%% If run from the command line make sure we quit MATLAB %%%%%%%%%%%%%%%%%%

quit();
