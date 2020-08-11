function [ Pg, P, A, pcA ] = loadPROFUMO( pfmDir, params )

% SPDX-License-Identifier: Apache-2.0

pfmDir = fullfile(pfmDir, 'FinalModel');

% Load group maps
mPg = h5read(fullfile(pfmDir, 'GroupSpatialModel', 'SignalMeans.post', ...
    'Means.hdf5'), '/dataset');
pPg = h5read(fullfile(pfmDir, 'GroupSpatialModel', 'Memberships.post', ...
    'Class_2', 'Probabilities.hdf5'), '/dataset');
Pg = mPg .* pPg; clear mPg pPg;

% And subject maps / timecourses
P = cell(params.S, 1);
A = cell(params.S, 1);
pcA = cell(params.S, 1);
for s = 1:params.S
    subj = sprintf('S%02d',s);
    subjDir = fullfile(pfmDir, 'Subjects', subj);
    
    % Subject maps
    mPs = h5read(fullfile(subjDir, 'SpatialMaps.post', 'Signal', ...
        'Means.hdf5'), '/dataset');
    pPs = h5read(fullfile(subjDir, 'SpatialMaps.post', 'Signal', ...
        'MembershipProbabilities.hdf5'), '/dataset');
    P{s} = mPs .* pPs; clear mPs pPs;
    
    % Time courses
    A{s} = cell(params.R(s), 1);
    pcA{s} = cell(params.R(s), 1);
    for r = 1:params.R(s)
        run = sprintf('R%02d',r);
        runDir = fullfile(subjDir, 'Runs', run);
        
        % Timecourses
        A{s}{r} = h5read(fullfile(runDir, 'TimeCourses.post', ...
            'CleanTimeCourses', 'Means.hdf5'), '/dataset');
            %'Means.hdf5'), '/dataset');
        % Modulate by amplitudes
        h = h5read(fullfile(runDir, 'ComponentWeightings.post', ...
            'Means.hdf5'), '/dataset');
        A{s}{r} = A{s}{r} .* h;
        
        % Temporal netmats
        precmat = h5read(fullfile(runDir, 'TemporalPrecisionMatrix.post', ...
            'Mean.hdf5'), '/dataset');
        pcA{s}{r} = - corrcov(precmat);
    end
end
