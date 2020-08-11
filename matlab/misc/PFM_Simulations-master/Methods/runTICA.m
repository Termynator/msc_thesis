function [ ticaP, ticaA ] = runTICA( svdU, svdS, svdV, params, nComps )
%Runs tICA on the temporally concatenated data set defined by the supplied
%SVD (data = [space x time])

% SPDX-License-Identifier: Apache-2.0

inds = 1:nComps; %Only take this many components from the SVD

%Concatenate SVD
svdVmat = zeros(params.T*sum(params.R), nComps);
sr = 1;
for s = 1:params.S
    for r = 1:params.R(s)
        svdVmat((sr-1)*params.T+(1:params.T), :) = svdV{s}{r}(:,inds);
        sr = sr + 1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V' = P*A  -->  D = (U*S*P) * A %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[icaA, icaP, ~] = fastica(svdVmat', ...
    'lastEig', nComps, 'approach', 'symm', 'verbose', 'off');

%Extract spatial maps
ticaP = svdU(:, inds) * svdS(inds, inds) * icaP;
%ticaP = repmat({ticaP}, params.S, 1);

%Extract time courses
ticaA = cell(params.S,1); sr = 1;
for s = 1:params.S
    ticaA{s} = cell(params.R(s),1);
    for r = 1:params.R(s)
        ticaA{s}{r} = icaA(:, (sr-1)*params.T+(1:params.T));
        sr = sr + 1;
    end
end

end
