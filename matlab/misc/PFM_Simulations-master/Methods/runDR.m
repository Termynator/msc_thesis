function [ P, A ] = runDR( D, Pg, params )
% Runs dual regression on the data set to give a set of subject specific maps
% and time courses

% SPDX-License-Identifier: Apache-2.0

nComps = size(Pg, 2);
EPS = 1.0e-5;

% Remove any almost zero maps to keep things more computationally stable
Pg(:, std(Pg) < EPS * max(std(Pg))) = 0.0;

% Regression for the time courses
iPg = pinv(Pg - mean(Pg,1));
A = cell(params.S,1);
for s = 1:params.S
    A{s} = cell(params.R(s),1);
    for r = 1:params.R(s)
        % Extract time courses
        Dsr = D{s}{r} - mean(D{s}{r},1);
        A{s}{r} = iPg * Dsr;
        
        % Remove any almost zero timecourses
        sA = std(A{s}{r}');
        A{s}{r}(sA < EPS * max(sA), :) = 0.0;
    end
end

% And for the subject maps
P = cell(params.S,1);
for s = 1:params.S
    % Extract data and time courses for this subject
    As = zeros(nComps, params.T*params.R(s));
    Ds = zeros(params.V, params.T*params.R(s));
    for r = 1:params.R(s)
        As(:, (r-1)*params.T+(1:params.T)) = A{s}{r} - mean(A{s}{r},2);
        Ds(:, (r-1)*params.T+(1:params.T)) = D{s}{r} - mean(D{s}{r},2);
    end
    % Pseudo-inverse to get P
    P{s} = Ds * pinv(As);
    
    % Remove any almost zero maps
    sP = std(P{s});
    P{s}(:, sP < EPS * max(sP)) = 0.0;
end

end
