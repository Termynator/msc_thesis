function [ scores ] = calculateDecompositionAccuracy( ...
        P, inf_P, A, inf_A, pcA, inf_pcA, params )
%CALCULATESCORES Returns accuracy of inferred maps and temporal
%correlations
%   Returns the correlation between the subject specific maps and time
%   courses, as well as the error between the spatial and temporal
%   correlation matrices (partial for temporal netmats)

% SPDX-License-Identifier: Apache-2.0

%--------------------------------------------------------------------------
% First calculate the raw correlations of maps / time courses to ground
% truth, and use that to pair components

%Calculate the map accuracies first
%Take cosine similarity between true and inferred maps
cP = 0;
for s = 1:params.S
    cP = cP + [P{s} inf_P{s}]' * [P{s} inf_P{s}];
end
%Normalise
cP = corrcov(cP);
%figure; imagesc(cP, [-1, 1]); colorbar
%Just the scores between the two
cP = cP( 1:params.N, params.N+(1:params.iN) );

%Repeat for the temporal accuracy
%Take correlation between true and inferred time courses
cA = 0;
for s = 1:params.S
    for r = 1:params.R(s)
        cA = cA + cov([A{s}{r}; inf_A{s}{r}]');
    end
end
%Normalise
cA = corrcov(cA);
%figure; imagesc(cA, [-1, 1]); colorbar
%Just the scores between the two
cA = cA( 1:params.N, params.N+(1:params.iN) );

%Match with the Hungarian algorithm and, record sign flips
[gt_inds, inf_inds] = linear_sum_assignment( -abs( cP + cA ) );
signs = diag(sign( cP(gt_inds,inf_inds) + cA(gt_inds,inf_inds) ));

%--------------------------------------------------------------------------

% Spatial accuracy - cosine similarity between subject maps
scores.P.data    = NaN(params.S, params.N);
scores.P.name    = 'Spatial maps';
scores.P.metric  = 'Cosine similarity';
scores.P.optimal = 1.0;
scores.P.range   = [-1.0, 1.0];
for s = 1:params.S
    % Just the scores between the two
    cP = cosine_sim(P{s}, inf_P{s});
    
    % Return scores in order of GT components so comparable across methods
    scores.P.data(s, gt_inds) = diag(cP(gt_inds, inf_inds)) .* signs;
end

%--------------------------------------------------------------------------

% Spatial accuracy - cross subject correlations
scores.P_xs.data    = NaN(params.V, params.N);
scores.P_xs.name    = 'Spatial maps (cross subject)';
scores.P_xs.metric  = 'Correlation';
scores.P_xs.optimal = 1.0;
scores.P_xs.range   = [-1.0, 1.0];
for n = 1:params.N
    Pn     = NaN(params.V, params.S);
    inf_Pn = NaN(params.V, params.S);
    for s = 1:params.S
        Pn(:,s) = P{s}(:, gt_inds(n));
        inf_Pn(:,s) = signs(n) .* inf_P{s}(:, inf_inds(n));
    end
    % Just the correlations between the two
    %cP = diag(corrcoef(Pn', inf_Pn'));  % [V, V]: Too big!
    dim = 2;
    zPn = zscore(Pn, 1, dim);  % normalise by N, not N-1
    inf_zPn = zscore(inf_Pn, 1, dim);
    cP = mean(zPn .* inf_zPn, dim);
    % Ignore elements with limited true variability
    cP(std(Pn, 1, dim) == 0.0) = NaN;
    cP(sum(Pn == 0.0, dim) > 0.2 * params.S) = NaN;
    
    % Return scores in order of GT components so comparable across methods
    scores.P_xs.data(:, gt_inds(n)) = cP;
end

%--------------------------------------------------------------------------

% Repeat for the temporal accuracy
% Take correlation between true and inferred time courses
scores.A.data    = NaN(sum(params.R), params.N);
scores.A.name    = 'Timecourses';
scores.A.metric  = 'Correlation';
scores.A.optimal = 1.0;
scores.A.range   = [-1.0, 1.0];
sr = 1;
for s = 1:params.S
    for r = 1:params.R(s)
        % Just the scores between the two
        cA = corr(A{s}{r}', inf_A{s}{r}');
        
        % Return scores in order of GT components so comparable across methods
        scores.A.data(sr, gt_inds) = diag(cA(gt_inds, inf_inds)) .* signs;
        
        sr = sr + 1;
    end
end

%--------------------------------------------------------------------------

% Accuaracy of amplitudes
% Assemble all elements into a [n, sr] matrix
aA = NaN(params.N, sum(params.R)); inf_aA = NaN(params.N, sum(params.R));
sr = 1;
for s = 1:params.S
    for r = 1:params.R(s)
        
        aA(:,sr) = std(A{s}{r}(gt_inds,:)');
        inf_aA(:,sr) = std(inf_A{s}{r}(inf_inds,:)');
        sr = sr + 1;
        
    end
end

% Save the correlations over amplitudes
% Each subject, across amplitudes (i.e. relative amplitudes similar)
% ** IGNORED ** Can't really disambiguate this from spatial scale
%scores.aA.data   = diag(corr(aA, inf_aA));
% Return scores in order of GT components so comparable across methods
%scores.aA.data    = scores.aA.data(gt_inds);
%scores.aA.name    = 'Amplitudes';
%scores.aA.metric  = 'Correlation';
%scores.aA.optimal = 1.0;
%scores.aA.range   = [-1.0, 1.0];
% Each element, across amplitudes (i.e. behavioural prediction)
scores.aA_xs.data    = diag(corr(aA', inf_aA'));
scores.aA_xs.name    = 'Amplitudes (cross subject)';
scores.aA_xs.metric  = 'Correlation';
scores.aA_xs.optimal = 1.0;
scores.aA_xs.range   = [-1.0, 1.0];

%figure; imagesc(aA, max(aA(:))*[-1 1]); colorbar();
%figure; imagesc(inf_aA, max(inf_aA(:))*[-1 1]); colorbar();
%figure; imagesc(zscore(aA')'); colorbar();
%figure; imagesc(zscore(inf_aA')'); colorbar();

%--------------------------------------------------------------------------

%Now find how well the mode interactions have been calculated

%First for the spatial correlations
N = min(params.N, params.iN); inds = (triu(ones(N),1) == 1);
% Assemble all off-diagonal elements into a [n, s] matrix
cPz = []; inf_cPz = [];
for s = 1:params.S
    
    %Find z-scored cosine similarity of real and observed maps
    cP_s = cosine_sim(P{s}(:,gt_inds));
    cPz = [cPz, r2z( cP_s(inds) )];
    
    inf_cP_s = (signs * signs') .* cosine_sim(inf_P{s}(:,inf_inds));
    inf_cPz = [inf_cPz, r2z( inf_cP_s(inds) )];
    
end

% Save the cosine similarity between unwrapped netmats
% Cosine sim as zero is meaningful for netmats
scores.cP.data    = diag(cosine_sim(cPz, inf_cPz));  % Each subject, across elements
scores.cP.name    = 'Spatial map interactions';
scores.cP.metric  = 'Cosine similarity';
scores.cP.optimal = 1.0;
scores.cP.range   = [-1.0, 1.0];
%figure; imagesc(cPz); colorbar
%figure; imagesc(inf_cPz); colorbar
%figure; imagesc(cosine_sim(cPz, inf_cPz), [-1, 1]); colorbar
%figure; imagesc(cosine_sim(cPz', inf_cPz'), [-1, 1]); colorbar

%--------------------------------------------------------------------------

%Now repeat for the temporal partial correlations
N = min(params.N, params.iN); inds = (triu(ones(N),1) == 1);
% Assemble all off-diagonal elements into a [n, sr] matrix
pcAz = []; inf_pcAz = [];
for s = 1:params.S
    for r = 1:params.R(s)
        
        pcA_sr = pcA{s}{r}(gt_inds, gt_inds);
        pcAz = [pcAz, r2z( pcA_sr(inds) )];
        inf_pcA_sr = (signs * signs') .* inf_pcA{s}{r}(inf_inds, inf_inds);
        inf_pcAz = [inf_pcAz, r2z( inf_pcA_sr(inds) )];
        
        %if (s == 1) && (r == 1)
        %    figure(); imagesc(pcA_sr, [-1 1]); colormap(bluewhitered); colorbar
        %    figure(); imagesc(inf_pcA_sr, [-1 1]); colormap(bluewhitered); colorbar
        %    figure(); plot(pcA_sr(:), inf_pcA_sr(:), '+');
        %    hold on; plot([-1 1], [-1 1]);
        %    xlim([-1 1]); ylim([-1 1]);
        %end
    end
end

%Save the RMS error between the correlation matrices
%pcAscore = (pcAz - inf_pcAz).^2;
%pcAscore = sqrt( mean(pcAscore)' );

% Save the correlations over netmat elements
% Each subject, across elements (i.e. are netmats similar)
% Cosine sim as zero is meaningful for netmats
scores.pcA.data    = diag(cosine_sim(pcAz, inf_pcAz));
scores.pcA.name    = 'Netmats';
scores.pcA.metric  = 'Cosine similarity';
scores.pcA.optimal = 1.0;
scores.pcA.range   = [-1.0, 1.0];
% Each element, across subjects (i.e. behavioural prediction)
scores.pcA_xs.data    = diag(corr(pcAz', inf_pcAz'));
scores.pcA_xs.name    = 'Netmats (cross subject)';
scores.pcA_xs.metric  = 'Correlation';
scores.pcA_xs.optimal = 1.0;
scores.pcA_xs.range   = [-1.0, 1.0];

%figure; imagesc(pcAz); colorbar
%figure; imagesc(inf_pcAz); colorbar
%figure; imagesc(cosine_sim(pcAz, inf_pcAz), [-1, 1]); colorbar
%figure; imagesc(corr(pcAz, inf_pcAz), [-1, 1]); colorbar
%figure; imagesc(corr(pcAz', inf_pcAz'), [-1, 1]); colorbar

%--------------------------------------------------------------------------

% And do the netmats bias one another (NAF2)?

% Repeat spatial correlations to get per-run metrics
cPz = repelem(cPz, 1, params.R);
inf_cPz = repelem(inf_cPz, 1, params.R);

% Each subject, across elements (i.e. are netmats similar)
% Cosine sim as zero is meaningful for netmats
scores.cP_pcA.data ...
    = diag(cosine_sim(inf_cPz, inf_pcAz)) - diag(cosine_sim(cPz, pcAz));
scores.cP_pcA.name    = 'Spatial-temporal netmat interactions';
scores.cP_pcA.metric  = 'Cosine similarity (difference from GT)';
scores.cP_pcA.optimal = 0.0;
scores.cP_pcA.range   = [-2.0, 2.0];
% Each element, across subjects (i.e. behavioural prediction)
scores.cP_pcA_xs.data ...
    = diag(corr(inf_cPz', inf_pcAz')) - diag(corr(cPz', pcAz'));
scores.cP_pcA_xs.name    = 'Spatial-temporal netmat interactions (cross subject)';
scores.cP_pcA_xs.metric  = 'Correlation (difference from GT)';
scores.cP_pcA_xs.optimal = 0.0;
scores.cP_pcA_xs.range   = [-2.0, 2.0];

%figure(); plot(cPz(:), pcAz(:), '.')
%figure(); plot(inf_cPz(:), inf_pcAz(:), '.')

%figure; imagesc(cPz, [-1 1]); colorbar
%figure; imagesc(inf_cPz, [-1 1]); colorbar
%figure; imagesc(pcAz, [-1 1]); colorbar
%figure; imagesc(inf_pcAz, [-1 1]); colorbar

%figure(); histogram(diag(corr(inf_cPz, inf_pcAz)) - diag(corr(cPz, pcAz)), 50)
%figure(); histogram(diag(corr(inf_cPz', inf_pcAz')) - diag(corr(cPz', pcAz')), 50)

%--------------------------------------------------------------------------

end
