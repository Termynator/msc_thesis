function [ sticaP, sticaA ] = runSTICA( D, svdU, svdS, svdV, params, nNodes, nModes )
%Runs sICA then tICA on the temporally concatenated data set defined by the
%supplied SVD (data = [space x time]). The respective dimensionalities of
%the two approaches are nNodes and nModes.

% SPDX-License-Identifier: Apache-2.0

%Run spatial ICA
[nodeP, nodeA] = runSICA(svdU, svdS, svdV, params, nNodes);

%Run dual regression to get subject specific versions
[nodeP_DR, nodeA_DR] = runDR(D, nodeA, params);

%Now normalise the timecourses and turn into an SVD form
vA = 0;
for s = 1:params.S
    for r = 1:params.R(s)
        vA = vA + var(nodeA_DR{s}{r}');
    end
end
vA = vA / sum(params.R);
nodeV = diag(sqrt(vA)); inodeV = diag(1./sqrt(vA));
for s = 1:params.S
    nodeP_DR{s} = nodeP_DR{s} * nodeV;
    for r = 1:params.R(s)
        nodeA_DR{s}{r} = inodeV * nodeA_DR{s}{r};
    end
end

%Re-run the SVD on the time courses to get into the right form for tICA
nodeParams = params; nodeParams.V = nNodes;
[~, ~, nodeSvdU, nodeSvdS, nodeSvdV] = runPCA(nodeA_DR, nodeParams, nModes);

%Run tICA
[modeP, modeA] = runTICA(nodeSvdU, nodeSvdS, nodeSvdV, params, nModes);

%Finally combine the mode and node results
sticaA = modeA;
sticaP = cell(params.S,1);
for s = 1:params.S
    sticaP{s} = nodeP_DR{s} * modeP;
end

end
