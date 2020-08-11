function [ D, plotHandle ] = generateData(PA, params, options, plotFigures)
%Generates noisy scans given observed BOLD signal

% SPDX-License-Identifier: Apache-2.0

if nargin == 3
    plotFigures = false;
end

%Decide what function to call based on what type of noise we want
switch options.D.noise.form
    
    case 'SpatiotemporallyWhiteGaussian'
        noise = generateData_STWhiteGaussian(PA, params, options, plotFigures);
        
    case 'SpatiotemporallyWhiteTdist'
        noise = generateData_STWhiteTdist(PA, params, options, plotFigures);
        
    case 'StructuredTdist'
        noise = generateData_StructuredTdist(PA, params, options, plotFigures);
        
    otherwise
        error('Not a recognised form for D')
        
end

% And then combine
D = combineData(PA, noise, params, options, plotFigures);

if plotFigures
    plotData(D, params, options);
end

plotHandle = @plotData;

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ D ] = combineData(PA, noise, params, options, plotFigures)
% Combines data and noise to match required SNR, with optional smoothing
%
% options.D.SNR - signal to noise ratio (expressed in terms of power)
% options.D.sigma_s - stddev of spatial Gaussian smoothing kernel
% options.D.sigma_t - stddev of temporal Gaussian smoothing kernel

% Generate smoothing kernels
if options.D.sigma_s > 0.0
    N = 2 * ceil(3 * options.D.sigma_s) + 1;
    alpha = (N - 1) / (2 * options.D.sigma_s);
    kernel_s = gausswin(N, alpha);
else
    kernel_s = [1.0];
end
if options.D.sigma_t > 0.0
    N = 2 * ceil(3 * options.D.sigma_t) + 1;
    alpha = (N - 1) / (2 * options.D.sigma_t);
    kernel_t = gausswin(N, alpha);
else
    kernel_t = [1.0];
end
kernel = kernel_s * kernel_t';
kernel = kernel / sum(kernel(:));
% Do the convolution
for s = 1:params.S
    for r = 1:params.R(s)
        PA{s}{r}    = conv2(PA{s}{r},    kernel, 'same');
        noise{s}{r} = conv2(noise{s}{r}, kernel, 'same');
    end
end

%Find global variance of BOLD signal
vPA = 0;
for s = 1:params.S
    for r = 1:params.R(s)
        vPA = vPA + var(PA{s}{r}(:));
    end
end
vPA = vPA / sum(params.R);

%Use SNR to find noise std from signal var
% SNR = var(signal) / var(noise)
noiseStd = sqrt( vPA / options.D.SNR );

%Add noise to the signal
D = cell(params.S, 1);
for s = 1:params.S
    D{s} = cell(params.R(s), 1);
    for r = 1:params.R(s)
        D{s}{r} = PA{s}{r} + noiseStd * noise{s}{r} / std(noise{s}{r}(:));
    end
end

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ noise ] = generateData_STWhiteGaussian(PA, params, options, plotFigures)
% Independent Gaussian noise

noise = cell(params.S, 1);
for s = 1:params.S
    noise{s} = cell(params.R(s), 1);
    for r = 1:params.R(s)
        noise{s}{r} = randn(params.V, params.T);
    end
end

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ noise ] = generateData_STWhiteTdist(PA, params, options, plotFigures)
% Independent t-distributed noise
%
% options.D.noise.a - gamma precision shape parameter
% options.D.noise.b - gamma precision rate parameter

%Plot probability distribution of precision parameters
if plotFigures
    gamMode = max(1, (options.D.noise.a-1)/options.D.noise.b);
    x = linspace(0, 3*gamMode, 500);
    
    figure; plot(x, gampdf(x, options.D.noise.a, 1/options.D.noise.b));
    xlabel('x'); xlim([x(1) x(end)]); ylabel('p(x)')
    title('t-distribution precisions')
end

% Generate noise
noise = cell(params.S, 1);
for s = 1:params.S
    noise{s} = cell(params.R(s), 1);
    for r = 1:params.R(s)
        precisions = gamrnd(options.D.noise.a, 1/options.D.noise.b, params.V, params.T);
        noise{s}{r} = randn(params.V, params.T) ./ sqrt(precisions);
    end
end

%Plot noise distribution
if plotFigures
    s = randi(params.S,1); r = randi(params.R(s),1);
    figure; histogram(noise{s}{r} / std(noise{s}{r}(:)), ...
        'Normalization', 'pdf');
    xlabel('Noise'); ylabel('Probility density');
    title(sprintf('Noise distribution (kurtosis: %.2f)', kurtosis(noise{s}{r}(:))))
end

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ noise ] = generateData_StructuredTdist(PA, params, options, plotFigures)
% Generates:
%  + a low rank structured subspace
%  + independent t-distributed noise
%
% options.D.noise.structuredSNR - structured / unstructured ('signal to noise') ratio (expressed in terms of power)
% options.D.noise.a - gamma precision shape parameter
% options.D.noise.b - gamma precision rate parameter
% options.D.noise.N - rank of structured noise

% SNR = var(signal) / var(noise)
structuredNoiseStd = sqrt( options.D.noise.structuredSNR );

% Global noise component mean (with some spatial structure)
n = 25; inds = cumsum(gamrnd(2.5, 1.0 / 2.5, [n, 1]));
inds = [1; round(params.V * inds / max(inds))];
vals = exprnd(1.0, [n, 1]) + 0.75 * randn([n, 1]);
vals = randn([n, 1]) + 0.5;
sPg = zeros([params.V, 1]);
for i = 1:n
    sPg(inds(i):inds(i+1)) = vals(i);
end
%sPg = exprnd(1.0, [params.V, 1]) - 0.5;

% Generate noise
noise = cell(params.S, 1);
for s = 1:params.S
    noise{s} = cell(params.R(s), 1);
    for r = 1:params.R(s)
        % Low rank subspace
        % Laplace maps and TCs, gamma amplitudes (and one global component too)
        sizeP = [params.V, options.D.noise.N];
        sP = exprnd(1.0, sizeP) .* sign(rand(sizeP) - 0.5);
        sP(:,1) = sPg + exprnd(1.0, [params.V, 1]);  % Global component
        sP = sP ./ sqrt(mean((sP.^2)));  % Normalise amplitudes
        
        sH = diag(gamrnd(5.0, 1.0 / 5.0, options.D.noise.N, 1));
        
        sizeA = [options.D.noise.N, params.T];
        sA = exprnd(1.0, sizeA) .* sign(rand(sizeA) - 0.5);
        
        structured = sP * sH * sA;
        
        
        % T-distribution, unstructured
        precisions = gamrnd(options.D.noise.a, 1/options.D.noise.b, params.V, params.T);
        unstructured = randn(params.V, params.T) ./ sqrt(precisions);
        
        
        % And combine
        noise{s}{r} = unstructured / std(unstructured(:)) ...
            + structuredNoiseStd * structured / std(structured(:));
    end
end

%Plot noise distribution
if plotFigures
    s = randi(params.S,1); r = randi(params.R(s),1);
    figure; histogram(noise{s}{r} / std(noise{s}{r}(:)), ...
        'Normalization', 'pdf');
    xlabel('Noise'); ylabel('Probility density');
    title(sprintf('Noise distribution (kurtosis: %.2f)', kurtosis(noise{s}{r}(:))))
    
    [U,S,V] = svd(noise{s}{r}, 0);
    figure(); plot(diag(S) / max(S(:)));
    ylim([-0.05 1.05]); ylabel('Relative singular value');
    title(['Example noise singular values: subject ' num2str(s) ', repeat ' num2str(r)])
    
    %figure; imagesc(noise{s}{r}, prctile(abs(noise{s}{r}(:)), 95.0)*[-1 1]); colorbar
    %title(['Example noise: subject ' num2str(s) ', repeat ' num2str(r)])
end

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ ] = plotData(D, params, options)

%Plot a couple of example scans
for i = 1:2
    s = randi(params.S,1); r = randi(params.R(s),1);
    
    figure; imagesc(D{s}{r}, prctile(abs(D{s}{r}(:)), 95.0)*[-1 1]); colormap('gray'); colorbar();
    title(['Example scan: subject ' num2str(s) ', repeat ' num2str(r)])
    
    %s = randi(params.S,1); r = randi(params.R(s),1);
    figure; imagesc(D{s}{r}, max(abs(D{s}{r}(:)))*[-1 1]); colormap('gray'); colorbar();
    title(['Example scan: subject ' num2str(s) ', repeat ' num2str(r)])
end

input('Press return to continue')
close all

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
