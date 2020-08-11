function [ An, plotHandle ] = generateNeuralTimecourses(params, options, plotFigures)
%Generates neural timecourses for fMRI simulations
%
% options.An.offRate - rate at each network is not present in subjects

% SPDX-License-Identifier: Apache-2.0

if nargin == 2
    plotFigures = false;
end

%Decide what function to call based on what type of timecourse we want
switch options.An.form
    
    case 'Freq'
        An = generateNeuralTimecourses_Freq(params, options, plotFigures);
        
    case 'CoupledHMM'
        An = generateNeuralTimecourses_CoupledHMM(params, options, plotFigures);
        
    otherwise
        error('Not a recognised form for P')
        
end

% Add amplitude variability
% And if the option has been specified, turn some networks off
for s = 1:params.S
    if isfield(options.An, 'offRate')
        on = (rand(params.N, 1) > options.An.offRate);
    else
        on = ones(params.N, 1);
    end
    for r = 1:params.R(s)
        amp = gamrnd( ...
            options.An.amp.a, 1.0 / options.An.amp.b, params.N, 1);
        An{s}{r} = An{s}{r} .* (amp .* on);
    end
end

%Finally, add a global rescaling such that all timecourses are overall
%unit variance
vA = 0;
for s = 1:params.S
    for r = 1:params.R(s)
        vA = vA + var(An{s}{r}(:));
    end
end
vA = vA / sum(params.R);
for s = 1:params.S
    for r = 1:params.R(s)
        An{s}{r} = An{s}{r} / sqrt(vA);
    end
end

if plotFigures
    plotNeuralTimecourses(An, params, options);
end

plotHandle = @plotNeuralTimecourses;

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ An ] = generateNeuralTimecourses_Freq(params, options, plotFigures)
%Generates a set of sparse timecourses with enhanced low frequency content
%
% options.An.Cg_dof - degrees of freedom of group Wishart correlation
% options.An.Cs_dof - degrees of freedom of subject Wishart correlation
% options.An.Cr_dof - degrees of freedom of run Wishart correlation
% options.An.fc - frequency cut off in Hz
% options.An.fAmp - amount to amplify frequencies less than fc by
% options.An.p - proportion of map entries to retain
% options.An.epsilon - std of additive noise

%Nyquist frequency
Fn = 1 / (2*params.dt);
%Frequencies available
f = linspace(0, Fn, floor(params.Tn/2)+1 );

%Generate global corr matrix
Cg = corrcov(wishrnd(eye(params.N) / options.An.Cg_dof, options.An.Cg_dof));

%Simulate the set of scans
An = cell(params.S, 1);
for s = 1:params.S
    
    %Generate subject corr matrix
    Cs = corrcov(wishrnd(eye(params.N) / options.An.Cs_dof, options.An.Cs_dof));
    
    An{s} = cell(params.R(s), 1);
    for r = 1:params.R(s)
        
        %Randomly generate frequency amplitudes
        fAn = abs(randn(params.N, length(f)));
        %Modulate frequencies below fc
        fAn(:, f < options.An.fc) = ...
            options.An.fAmp * fAn(:, f < options.An.fc);
        
        %Add random phase
        fAn = fAn .* exp( 2*pi*rand(size(fAn))*sqrt(-1) );
        %But DC component should be real
        fAn(:,1) = abs( fAn(:,1) );
        
        
        %Add conjugate 'high f' content for FFT
        %This changes slightly depending on whether we have an odd or even
        %number of time points to simulate
        if mod(params.Tn, 2) == 0
            %Case for even Tn
            fAn(:,end) = abs( fAn(:,end) );
            fAn = [fAn conj(fAn(:, end-1:-1:2))];
        else
            %Odd Tn
            fAn = [fAn conj(fAn(:, end:-1:2))];
        end
        
        %Now invert - we should get a real time course back
        An{s}{r} = ifft( fAn' )';
        
        %Generate run corr matrix
        Cr = corrcov(wishrnd(eye(params.N) / options.An.Cr_dof, options.An.Cr_dof));
        % Combine together
        C = Cr^0.5 * Cs^0.5 * Cg * Cs^0.5 * Cr^0.5;
        %figure(); imagesc(corrcov(C), [-1 1]); colorbar()
        %Rotate to induce correlations
        An{s}{r} = C^0.5 * An{s}{r};  % * (An{s}{r} * An{s}{r}')^-0.5
        %figure(); imagesc(An{s}{r}); colorbar()
        %figure(); imagesc(corr(An{s}{r}'), [-1 1]); colorbar()
        
        % Normalise amplitudes: mixing to induce correlations can change these
        An{s}{r} = An{s}{r} ./ std(An{s}{r}')';
        
        %Assume timecourses are Gaussian and find expected value of
        %percentile given by p
        mu = norminv([options.An.p/2 1-options.An.p/2], ...
            0, std(An{s}{r}(:)));
        %Sparsify
        An{s}{r}( (An{s}{r} > mu(1)) & (An{s}{r} < mu(2)) ) = 0;
        
    end
end


%Rescale to unit variance and add some noise
vA = 0;
for s = 1:params.S
    for r = 1:params.R(s)
        vA = vA + var(An{s}{r}(:));
    end
end
vA = vA / sum(params.R);
for s = 1:params.S
    for r = 1:params.R(s)
        An{s}{r} = An{s}{r} / sqrt(vA);
        An{s}{r} = An{s}{r} ...
            + options.An.epsilon * randn(params.N, params.Tn);
    end
end

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [  ] = plotNeuralTimecourses(An, params, options)

%Plot a couple of time courses from a scan
s = randi(params.S,1); r = randi(params.R(s),1);
t = params.dt * (0:(params.Tn-1));
figure; plot(t, An{s}{r}(randi(params.N,1),:), 'Color', 0.8*[1 1 1])
hold on; plot(t, An{s}{r}(randi(params.N,1),:));
xlim([0 50]); xlabel('Time');
set(gca, 'YTick', 0); ylabel('Neural signal (a.u.)')
title(['Example neural time courses: subject ' num2str(s) ', repeat ' num2str(r)])

%Plot frequency content
NFFT = 2^nextpow2(params.Tn);
Fs = 1/params.dt; %sampling frequency
f = Fs/2*linspace(0,1,NFFT/2+1);
% mean frequency content for each scan
figure; hold on; fAn = 0;
for s = 1:params.S
    for r = 1:params.R(s)
        fAn = fAn + mean(abs(fft(An{s}{r}', NFFT)'));
    end
end
fAn = fAn / sum(params.R);
plot(f, fAn(1:NFFT/2+1), 'b');
xlim([0 Fs/2]); xlabel('Frequency (Hz)')
set(gca, 'YTick', 0); ylabel('FFT')
title('Frequency content of neuronal signal')

%Plot the distribution of values
figure; histogram(An{s}{r}(:), 'Normalization', 'pdf');
xlabel('Neural signal (a.u.)'); ylabel('Probility density');
title('Distribution of neural activity')

%Look at temporal correlations
cAn = 0;
for s = 1:params.S
    for r = 1:params.R(s)
        cAn = cAn + An{s}{r} * An{s}{r}';
    end
end
dcA = sqrt(diag(cAn));
cAn = cAn ./ (dcA * dcA');
figure; imagesc(cAn, [-1 1]); colorbar; colormap(bluewhitered)
title('Global temporal correlations')

% As = [];
% for s = 1:params.S
%     A = [];
%     for r = 1:params.R(s)
%         A = [A An{s}{r}];
%     end
%     As = [As; A];
% end
% cAs = As*As';
% dcA = 1./sqrt(diag(cAs));
% cAs = bsxfun(@times, cAs, dcA);
% cAs = bsxfun(@times, cAs, dcA');
% figure; imagesc(cAs, [-1 1]); colorbar; colormap(bluewhitered)
%
% Ar = [];
% for s = 1:params.S
%     for r = 1:params.R(s)
%         Ar = [Ar; An{s}{r}];
%     end
% end
% cAr = Ar*Ar';
% dcA = 1./sqrt(diag(cAr));
% cAr = bsxfun(@times, cAr, dcA);
% cAr = bsxfun(@times, cAr, dcA');
% figure; imagesc(cAr, [-1 1]); colorbar; colormap(bluewhitered)

input('Press return to continue')
close all

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
