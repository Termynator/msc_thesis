function [ P, A, Pt, At ] = loadDualRegression( dualregDir, params )

% SPDX-License-Identifier: Apache-2.0

% Subject maps / timecourses
% And thresholded versions too
P  = cell(params.S, 1);
Pt = cell(params.S, 1);
A  = cell(params.S, 1);
At = cell(params.S, 1);
sr = 0;
for s = 1:params.S
    P{s}  = 0.0;
    Pt{s} = 0.0;
    A{s}  = cell(params.R(s), 1);
    At{s} = cell(params.R(s), 1);
    for r = 1:params.R(s)
        subj = sprintf('subject%05d', sr);
        
        % Collate run-specific maps
        Psr = read_avw(fullfile( ...
            dualregDir, ['dr_stage2_' subj '.nii.gz']));
        P{s} = P{s} + reshape(Psr, params.V, params.iN);
        Psr = read_avw(fullfile( ...
            dualregDir, ['dr_stage4_' subj '_thresh.nii.gz']));
        Pt{s} = Pt{s} + reshape(Psr, params.V, params.iN);
        
        % Timecourses
        A{s}{r} = load(fullfile( ...
            dualregDir, ['dr_stage1_' subj '.txt']))';
        At{s}{r} = load(fullfile( ...
            dualregDir, ['dr_stage4_' subj '.txt']))';
        
        sr = sr + 1;
    end

    % Take mean map
    P{s}  = P{s} ./ params.R(s);
    Pt{s} = Pt{s} ./ params.R(s);
end

end
