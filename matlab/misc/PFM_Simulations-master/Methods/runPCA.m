function  [ pcaP, pcaA, svdU, svdS, svdV ] = runPCA(D, params, nComps)
%Runs PCA on the temporally concatenated data set D
%   Also returns the full SVD

% SPDX-License-Identifier: Apache-2.0

%Concatenate the data
Dmat = zeros(params.V, params.T*sum(params.R));
sr = 1;
for s = 1:params.S
    for r = 1:params.R(s)
        Dmat(:, (sr-1)*params.T+(1:params.T)) = D{s}{r};
        sr = sr + 1;
    end
end

%Run the svd
%%%%%%%%%%%%%%
% D = U*S*V' %
%%%%%%%%%%%%%%
[svdU, svdS, svdVmat] = svd(Dmat, 'econ');

%For the PCA itself, take first nComps
inds = 1:nComps;
pcaAmat = svdVmat(:, inds)';
pcaP = svdU(:, inds) * svdS(inds, inds);
%pcaP = repmat({pcaP}, params.S, 1);

%Put matrices back in subject / repeat cell form
sr = 1; pcaA = cell(params.S,1); svdV = cell(params.S,1);
for s = 1:params.S
    pcaA{s} = cell(params.R(s),1);
    svdV{s} = cell(params.R(s),1);
    for r = 1:params.R(s)
        pcaA{s}{r} = pcaAmat(:, (sr-1)*params.T+(1:params.T));
        svdV{s}{r} = svdVmat((sr-1)*params.T+(1:params.T), :);
        sr = sr + 1;
    end
end

end
