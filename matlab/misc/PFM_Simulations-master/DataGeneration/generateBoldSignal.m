function [ PA, A, plotHandle ] = generateBoldSignal(P, An, params, options, plotFigures)
%Generates BOLD signal for fMRI simulations
%  Takes spatial maps and neuronal timecourses and gives the downsampled
%  BOLD signal and timecourses

% SPDX-License-Identifier: Apache-2.0

if nargin == 4
    plotFigures = false;
end

%Decide what function to call based on what type of timecourse we want
switch options.BS.form
    
    case 'LinearHRF'
        [PA, A] = generateBoldSignal_LinearHRF(P, An, params, options, plotFigures);
        
    case 'FlobsHRF'
        [PA, A] = generateBoldSignal_FlobsHRF(P, An, params, options, plotFigures);
        
    case 'SaturatingFlobsHRF'
        [PA, A] = generateBoldSignal_SaturatingFlobsHRF(P, An, params, options, plotFigures);
        
    case 'BalloonModelHRF'
        [PA, A] = generateBoldSignal_BalloonModelHRF(P, An, params, options, plotFigures);
        
    otherwise
        error('Not a recognised form for PA')
        
end

%Finally, add a global rescaling such that all timecourses are overall
%unit variance
vPA = 0;
for s = 1:params.S
    for r = 1:params.R(s)
        vPA = vPA + var(PA{s}{r}(:));
    end
end
vPA = vPA / sum(params.R);
for s = 1:params.S
    for r = 1:params.R(s)
        PA{s}{r} = PA{s}{r} / sqrt(vPA);
    end
end

if plotFigures
    plotBoldSignal(PA, P, An, params, options);
end

plotHandle = @plotBoldSignal;

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ HRF, t_HRF ] = loadFLOBS(dt_HRF)
% Loads the FLOBS HRF basis, and resamples to `dt_HRF`
% Optionally returns the sample times, `t_HRF`

% Load the basis
fsldir = getenv('FSLDIR');
assert(~isempty(fsldir), 'Could not find `$FSLDIR`!');
hrf = load(fullfile(fsldir, 'etc', 'default_flobs.flobs', 'hrfbasisfns.txt'));
dt_hrf = 0.05;   %FLOBS sampling rate

% Establish time samplings
n_hrf = length(hrf);
t_hrf = linspace(0.0, dt_hrf * (n_hrf - 1), n_hrf);
n_HRF = floor(t_hrf(end) / dt_HRF);
t_HRF = linspace(0.0, dt_HRF * (n_HRF - 1), n_HRF);

% Interpolate HRF to requested rate
HRF = interp1(t_hrf, hrf, t_HRF, 'pchip', 0.0);

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ PA, A ] = generateBoldSignal_LinearHRF(P, An, params, options, plotFigures)
%Convolves timecourses with the mean FLOBS linear HRF and mixes with spatial
%maps to get BOLD signal
%
% options.BS.SNR_P - SNR (in terms of power) of the maps
% options.BS.SNR_A - SNR (in terms of power) of the timecourses

%Because everything is linear we can convolve the network, rather than
%voxelwise timecourses

%Load the HRF
hrf = loadFLOBS(params.dt); hrf = hrf(:,1);

PA = cell(params.S,1);
A = cell(params.S,1);
for s = 1:params.S
    
    PA{s} = cell(params.R(s),1);
    A{s} = cell(params.R(s),1);
    for r = 1:params.R(s)
        
        A{s}{r} = zeros(params.N, params.T);
        
        %Loop over networks, generating timecourses
        for n = 1:params.N
            
            %Convolve with the HRF, only taking the non-zero padded times
            An_hrf = conv(An{s}{r}(n,:), hrf, 'valid');
            
            %Generate the timestamps for both
            tn = params.dt * (1:length(An_hrf));
            t = params.TR * (1:params.T);
            %Take the end of the timecourse
            t = t + tn(end) - t(end);
            %Downsample at TR
            A{s}{r}(n,:) = interp1(tn, An_hrf, t, 'pchip');
            
        end
        
        %Add some noise and multiply by spatial maps to get voxelwise signal
        p = P{s}; vp = mean(p(:).^2);
        p = p + sqrt(vp / options.BS.SNR_P) * randn(size(p));
        a = A{s}{r}; va = mean(a(:).^2);
        a = a + sqrt(va / options.BS.SNR_A) * randn(size(a));
        PA{s}{r} = p * a;
        
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ PA, A ] = generateBoldSignal_FlobsHRF(P, An, params, options, plotFigures)
%Convolves timecourses with a random draw from the FLOBS HRF basis set and
%mixes with spatial maps to get BOLD signal
%
% options.BS.HRFcoeffs.mu - mean of the value of the 3 FLOBS coefficients
% options.BS.HRFcoeffs.sigma - std of the value of the 3 FLOBS coefficients
% options.BS.SNR_P - SNR (in terms of power) of the maps
% options.BS.SNR_A - SNR (in terms of power) of the timecourses

%Because everything is linear we can convolve the network, rather than
%voxelwise timecourses

%Load the HRF
[hrf, t_hrf] = loadFLOBS(params.dt);

if plotFigures
    figure; hold all; nPlots = 0;
    xlim([t_hrf(1) t_hrf(end)]); xlabel('Time'); ylabel('HRF')
    title('FLOBS HRF: example HRFs')
end

PA = cell(params.S,1);
A = cell(params.S,1);
for s = 1:params.S
    
    %Generate a random HRF for each subjects networks
    randHrf = zeros(params.N, length(hrf));
    for n = 1:params.N
        %Generate the HRF according to options
        for i = 1:3
            randHrf(n,:) = randHrf(n,:) + ...
                hrf(:,i)' * ( options.BS.HRFcoeffs.mu(i) ...
                + options.BS.HRFcoeffs.sigma(i)*randn() );
        end
    end
    if plotFigures && (nPlots<100)
        plot(t_hrf, randHrf); nPlots = nPlots + params.N;
    end
    
    PA{s} = cell(params.R(s),1);
    A{s} = cell(params.R(s),1);
    for r = 1:params.R(s)
        
        A{s}{r} = zeros(params.N, params.T);
        
        %Loop over networks, generating timecourses
        for n = 1:params.N
            
            %Convolve with the HRF, only taking the non-zero padded times
            An_hrf = conv(An{s}{r}(n,:), randHrf(n,:), 'valid');
            
            %Generate the timestamps for both
            tn = params.dt * (1:length(An_hrf));
            t = params.TR * (1:params.T);
            %Take the end of the timecourse
            t = t + tn(end) - t(end);
            %Downsample at TR
            A{s}{r}(n,:) = interp1(tn, An_hrf, t, 'pchip');
            
        end
        
        %Add some noise and multiply by spatial maps to get voxelwise signal
        p = P{s}; vp = mean(p(:).^2);
        p = p + sqrt(vp / options.BS.SNR_P) * randn(size(p));
        a = A{s}{r}; va = mean(a(:).^2);
        a = a + sqrt(va / options.BS.SNR_A) * randn(size(a));
        PA{s}{r} = p * a;
        
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ PA, A ] = generateBoldSignal_SaturatingFlobsHRF(P, An, params, options, plotFigures)
%Convolves timecourses with a random draw from the FLOBS HRF basis set and
%mixes with spatial maps to get BOLD signal. Then saturates signal by
%subtracting a portion of te square of the signal (very simple approx to a
%Volterra kernel)
%
% options.BS.HRFcoeffs.mu - mean of the value of the 3 FLOBS coefficients
% options.BS.HRFcoeffs.sigma - std of the value of the 3 FLOBS coefficients
% options.BS.tanhPercentile - what percentile of BOLD value to normalise by
% options.BS.tanhMax - where the maximum BOLD value gets mapped to in tanh
% options.BS.SNR_P - SNR (in terms of power) of the maps
% options.BS.SNR_A - SNR (in terms of power) of the timecourses

%First generate the linear signal
[ PA, A ] = generateBoldSignal_FlobsHRF(P, An, params, options, plotFigures);

if plotFigures
    %Plot the tanh nonlinearity
    x = linspace(-2.0*options.BS.tanhMax, 2.0*options.BS.tanhMax, 250);
    figure; hold on
    hLin = plot([-1 1],[-1 1],'r:');
    hLims = plot(options.BS.tanhMax*[1 1] ,[-1 1] ,'--', 'Color', 0.7*[1 1 1]);
    plot(-options.BS.tanhMax*[1 1], [-1 1], '--', 'Color', 0.7*[1 1 1]);
    hTanh = plot(x, tanh(x));
    plot(0,0,'r+','MarkerSize',15)
    legend([hTanh, hLin, hLims], {'Nonlinearity', 'Linear', 'Signal lims'}, 'Location', 'NorthWest')
    xlim([x(1) x(end)]); xlabel('Signal in')
    ylim([-1 1]); xlabel('Signal out')
    
    signalLoss = 100 * (options.BS.tanhMax - tanh(options.BS.tanhMax)) ...
        / options.BS.tanhMax;
    title(sprintf('HRF tanh nonlinearity (signal loss %d%%)', round(signalLoss)))
end

%Then subtract areas where signal is large
for s = 1:params.S
    for r = 1:params.R(s)
        
        %Normalise scan
        mPA = prctile(abs((PA{s}{r}(:))), options.BS.tanhPercentile);
        PA{s}{r} = PA{s}{r} / mPA;
        
        if plotFigures && (s==1) && (r==1)
            figure; histogram(PA{s}{r}(:), linspace(-1.2, 1.2, 50), ...
                'Normalization', 'pdf'); xlim(1.2*[-1 1]);
            xlabel('Kurtosis'); ylabel('Probility density');
            title(sprintf('PA distribution pre-saturation (kurtosis: %.2f)', ...
                kurtosis(PA{s}{r}(:))))
            
            ind = max(abs(PA{s}{r}')); [~,ind] = max(ind);
            t = params.TR * (0:(params.T-1));
            figure; plot(t, PA{s}{r}(ind,:))
        end
        
        %Subtract a proportion of the square of the signal
        % options.BS.satCoeff - proportion to subtract off
        %PA{s}{r} = PA{s}{r} ...
        %    - options.BS.satCoeff * (sign(PA{s}{r}) .* PA{s}{r}.^2);
        
        %Pass through a tanh nonlinearity
        PA{s}{r} = tanh( options.BS.tanhMax * PA{s}{r} ) / options.BS.tanhMax;
        
        if plotFigures && (s==1) && (r==1)
            hold on; plot(t, PA{s}{r}(ind,:), 'r:')
            xlim([t(1) t(end)]); xlabel('Time')
            set(gca, 'YTick', 0); ylabel('BOLD signal (a.u.)')
            legend('Linear', 'Saturating')
            title('Saturating HRF: Comparison with linear HRF')
            
            figure; histogram(PA{s}{r}(:), linspace(-1.2, 1.2, 50), ...
                'Normalization', 'pdf'); xlim(1.2*[-1 1]);
            xlabel('Kurtosis'); ylabel('Probility density');
            title(sprintf('PA distribution post-saturation (kurtosis: %.2f)', ...
                kurtosis(PA{s}{r}(:))))
        end
    end
end

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ ] = plotBoldSignal(PA, P, An, params, options)

%Plot an example scan
s = randi(params.S,1); r = randi(params.R(s),1);
figure; imagesc(PA{s}{r}, 6*[-1 1]); colormap('gray'); colorbar();
title(['Example scan: subject ' num2str(s) ', repeat ' num2str(r)])

%Plot a couple of timecourses from that scan
t = params.TR * (0:(params.T-1));
figure; plot(t, PA{s}{r}(randi(params.V,1),:), 'Color', 0.8*[1 1 1])
hold on; plot(t, PA{s}{r}(randi(params.V,1),:));
xlim([0 300]); xlabel('Time');
set(gca, 'YTick', 0); ylabel('BOLD signal (a.u.)')
title(['Example time courses: subject ' num2str(s) ', repeat ' num2str(r)])

%Plot frequency content
NFFT = params.T;
Fs = 1/params.TR; %sampling frequency
f = Fs/2*linspace(0,1,NFFT/2+1);
%Example HRF
hrf = loadFLOBS(params.TR); hrf = hrf(:,1);
fHRF = abs(fft(hrf, NFFT))';
% mean frequency content for each scan
figure; hold on
fPA = 0;
for s = 1:params.S
    for r = 1:params.R(s)
        fTemp = mean(abs(fft(PA{s}{r}', NFFT)'));
        fPA = fPA + fTemp;
        
        hTemp = plot(f, fTemp(1:NFFT/2+1), 'Color', 0.8*[1 1 1]);
    end
end
fPA = fPA / sum(params.R);
hPA = plot(f, fPA(1:NFFT/2+1), 'b');
hHRF = plot(f, (fPA / fHRF) * fHRF(1:NFFT/2+1), 'r--');
xlim([0 Fs/2]); xlabel('Frequency (Hz)')
set(gca, 'YTick', 0); ylabel('FFT')
legend([hTemp, hPA, hHRF], 'Mean scan FFTs', 'Mean of scan FFTs', 'Example HRF FFT')
title('Frequency content of BOLD signal')


%Examine best guess at node time courses

%Use the subject specific maps to regress the timecourses out of the BOLD signal
A = cell(params.S,1);
for s = 1:params.S
    A{s} = cell(params.R(s),1);
    for r = 1:params.R(s)
        A{s}{r} = P{s} \ PA{s}{r};
    end
end

%Extract neural and BOLD correlations from one scan
s = randi(params.S,1); r = randi(params.R(s),1);
cA = corrcoef(A{s}{r}'); cA(isnan(cA)) = 0;
cAn = corrcoef(An{s}{r}'); cAn(isnan(cAn)) = 0;

%Plot correlation matrices
figure; imagesc(cA, [-1 1]); colorbar; colormap(bluewhitered)
title(['BOLD correlations, example scan: subject ' num2str(s) ', repeat ' num2str(r)])
figure; imagesc(cAn, [-1 1]); colorbar; colormap(bluewhitered)
title(['Neural correlations, example scan: subject ' num2str(s) ', repeat ' num2str(r)])
%Plot how the two relate
figure; plot([-1 1], [-1 1], 'r--'); hold on; plot(cAn(:), cA(:), '.')
axis square; xlim([-1 1]); ylim([-1 1])
xlabel('Neural correlation'); ylabel('BOLD correlation')
title(['Example scan: subject ' num2str(s) ', repeat ' num2str(r)])

cA = 0; cAn = 0;
for s = 1:params.S
    for r = 1:params.R(s)
        cA = cA + A{s}{r} * A{s}{r}';
        cAn = cAn + An{s}{r} * An{s}{r}';
    end
end
dcA = sqrt(diag(cA));
cA = cA ./ (dcA * dcA');
dcAn = sqrt(diag(cAn));
cAn = cAn ./ (dcAn * dcAn');

%Plot correlation matrices
figure; imagesc(cA, [-1 1]); colorbar; colormap(bluewhitered)
title('Global BOLD correlations')
figure; imagesc(cAn, [-1 1]); colorbar; colormap(bluewhitered)
title('Global neural correlations')
%Plot how the two relate
figure; plot([-1 1], [-1 1], 'r--'); hold on; plot(cAn(:), cA(:), '.')
axis square; xlim([-1 1]); ylim([-1 1])
xlabel('Neural correlation'); ylabel('BOLD correlation')
title('Global')


input('Press return to continue')
close all

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
