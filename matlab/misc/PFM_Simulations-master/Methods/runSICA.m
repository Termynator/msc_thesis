function [ sicaP, sicaA ] = runSICA( svdU, svdS, svdV, params, nComps )
%Runs sICA on the temporally concatenated data set defined by the supplied
%SVD (data = [space x time])

% SPDX-License-Identifier: Apache-2.0

inds = 1:nComps; %Only take this many components from the SVD

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% U' = P*A  -->  D = A'*(P'*S*V') %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Run the ICA on the spatial dimension
[icaA, icaP, ~] = fastica(svdU(:, inds)', ...
    'lastEig', nComps, 'approach', 'symm', 'verbose', 'off');

%Extract spatial maps
sicaP = icaA';
%sicaP = repmat({sicaP}, params.S, 1);

%Extract time courses
sicaA = cell(params.S,1);
for s = 1:params.S
    sicaA{s} = cell(params.R(s),1);
    for r = 1:params.R(s)
        sicaA{s}{r} = icaP' * svdS(inds, inds) * svdV{s}{r}(:, inds)';
    end
end

end
