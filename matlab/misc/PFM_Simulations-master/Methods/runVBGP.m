function [ P, A, Pg ] = runVBGP( D, params, nComps, Pgt, Agt )
%Runs VBGP on the supplied data set
%   Returns the posterior mean maps and time courses

% SPDX-License-Identifier: Apache-2.0

%% Set the prior

initVBGP = 'rand'; %'randDualReg';
max_iter = 2500;
tolerance = 5; %-Inf;

% P
prior.P.form = 'DistSASS';
%Scale voxel priors by number of subjects
prior.P.q.spike = (params.S/10) * 0.45;
prior.P.q.slab = (params.S/10) * 0.05;
prior.P.alpha.a = (params.S/10) * 1;
prior.P.alpha.b = (params.S/10) * 1;
%Scale global sparsity by number of networks
%prior.P.mu.p = 1.5 / params.N;
prior.P.mu.p = 0.1;
%Uninformative prior over global scalings
prior.P.mu.alpha.b = 1;
prior.P.mu.alpha.a = 1;

% A
prior.A.form = 'HierarchicalFull';
[Khrf, hrf] = generateTemporalPrior( 'HRF', params.T, params.TR, false );
prior.A.sigma = {Khrf};
prior.A.hrf = {hrf};
%Tight prior on scalings - want networks to have HRF properties
prior.alpha.a = 0.5 * (params.T*params.N*sum(params.R)/2);
prior.alpha.b = 0.5 * (params.T*params.N*sum(params.R)/2);
prior.beta.a = 1e2 * (params.T*params.N*mean(params.R)/2);
prior.beta.b = 1e2 * (params.T*params.N*mean(params.R)/2);

% mu
prior.mu.nu = 0.1;

% psi
prior.psi.a = 1;
prior.psi.b = 1;

%%
prior.P.form = 'Hierarchical';
prior.Pg.form = 'SS';
prior.Pg.p = 0.1;
prior.Pg.R.alpha.a = 1;
prior.Pg.R.alpha.b = 1;
prior.Phi.form = 'Null';

%% Run algorithm

if nargin == 5
    
    [posterior, F] = HierarchicalFactoredVBGP(D, prior, ...
        'Initialisation', initVBGP, 'numComp', nComps, ...
        'maxIter', max_iter, 'tolerance', tolerance, 'econ', ...
        'normalisedData', 'groundTruth', Pgt, Agt);
    
elseif nargin == 3
    
    %Just run VBGP normally
    [posterior, F] = HierarchicalFactoredVBGP(D, prior, ...
        'Initialisation', initVBGP, 'numComp', nComps, ...
        'maxIter', max_iter, 'tolerance', tolerance, 'econ', ...
        'normalisedData');
end

%% Extract results

P = cell(params.S,1); A = cell(params.S,1);
for s = 1:params.S
    P{s} = posterior.P.R.m{s} .* posterior.P.S{s};
    for r = 1:params.R(s)
        A{s}{r} = posterior.A.m{s}{r}{1};
    end
end

Pg = posterior.P.mu.R.m .* posterior.P.mu.S;

end
