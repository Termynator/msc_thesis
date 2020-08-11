function [  ] = plotScores( scores, params, outputDir, plotFigures )
%Plots the results from whichever analyses have been run

% SPDX-License-Identifier: Apache-2.0

if nargin < 3
    outputDir = false;
end
saveFigures = logical(outputDir);

if nargin < 4
    plotFigures = true;
end
if plotFigures
    visibility = 'on';
else
    visibility = 'off';
end

% https://github.com/altmany/export_fig/issues/75
warning('off', 'MATLAB:prnRenderer:opengl');
warning('off', 'export_fig:transparency');

%--------------------------------------------------------------------------
% Get all the metadata sorted

methods = fieldnames(scores)

tests = struct2cell(structfun(@fieldnames, scores, 'UniformOutput', false));
tests = unique(vertcat(tests{:}))

if saveFigures
    mkdir(outputDir);
    % [left bottom width height]
    figPosLong = 200 + [0 0 1000 200];
    figPosSquare = 200 + [0 0 300 300];
end
ticks = 0:0.2:1; %Where to place ticks for correlations

%--------------------------------------------------------------------------
% Plot all the tests

for test = tests'
    test = test{1};
    
    results = combineResults(scores, params, test);
    
    % Plot marginals
    plotDistributions( ...
        results.methods, results.data, results.optimal, results.range, ...
        visibility);
    ylabel(results.metric);
    title(results.name);
    if saveFigures
       set(gcf(), 'Position', figPosSquare);
       set(gcf(), 'Color', 'w');
       export_fig( ...
           fullfile(outputDir, strrep(results.name, ' ', '_')), ...
           '-pdf', '-painters')  % '-transparent'
    end
    
    % And all data
    plotAllVals( ...
        results.methods, results.data, results.optimal, results.range, ...
        visibility);
    ylabel(results.metric);
    title(results.name);
    if saveFigures
       set(gcf(), 'Position', figPosLong);
       set(gcf(), 'Color', 'w');
       export_fig( ...
           fullfile(outputDir, [strrep(results.name, ' ', '_') '_all']), ...
           '-pdf', '-painters')  % '-transparent'
    end
end

% Tidy up invisible figures that have been saved
%if ~plotFigures && saveFigures
%    close all
%end

%--------------------------------------------------------------------------
if false

input('Press return to continue')

%--------------------------------------------------------------------------
% Plot the test-retest scores

%Plot correlation results (Ptr)
[PTRmethods, PTRresults] = extractResults(scores, params, 'Ptr');

plotDistributions( PTRmethods, PTRresults );
ylim([0 1]+0.05*[-1 1]); set(gca, 'YTick', ticks);
ylabel('Correlation coefficient')
if saveFigures
    set(gcf, 'Position', figPosLong)
    export_fig('SimData_SpatialMapTR', '-pdf', '-transparent')
end
title('Spatial Map Repeatability')

%--------------------------------------------------------------------------
% Plot the relationships between map and time course accuracy

range.x = [0 1]; range.y = [0 1];
H = makeScatterPlots(Pmethods, Presults, Amethods, Aresults, range);
for n = 1:length(H)
    set(0, 'CurrentFigure', H(n))
    axis square
    xlabel('Map accuracy')
    ylabel('Time course accuracy')
    set(gca, 'XTick', ticks); set(gca, 'YTick', ticks)
end

input('Press return to continue')
close(H)

%--------------------------------------------------------------------------
% Plot the relationships between spatial and temporal correlation accuracy

range.x = [0 0.7]; range.y = [0 0.7];
H = makeScatterPlots(cPmethods, cPresults, cAmethods, cAresults, range);
for n = 1:length(H)
    set(0, 'CurrentFigure', H(n))
    axis square
    xlabel('Spatial correlation accuracy')
    ylabel('Temporal correlation accuracy')
end

input('Press return to continue')
close(H)

%--------------------------------------------------------------------------
% Plot the relationship between map accuracy and test-retest

H = makeAccuracyRetestPlots(PTRmethods, PTRresults, Pmethods, Presults);
for n = 1:length(H)
    figure(H(n))
    set(0, 'CurrentFigure', H(n))
    set(gca, 'XTick', ticks); set(gca, 'YTick', ticks)
    if saveFigures
        method = get(gca,'Title'); method = get(method,'String'); title('')
        set(gcf, 'Position', figPosSquare)
        export_fig(['SimData_Acc-v-TR_' method], '-pdf', '-transparent')
        title(method)
    end
end

end
%--------------------------------------------------------------------------

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ results ] = combineResults( scores, params, testName )
%Extracts the set of scores for a given test from the 'scores' structure
%   Returns a list of the methods with that score, as well as the scores
%   themselves

results = struct();

%Extract the methods that have been tested
results.methods = fieldnames(scores);
results.name    = {};
results.metric  = {};
results.optimal = [];
results.range   = [];

%Loop over methods, extracting scores where appropriate
n = 1;
while n <= numel(results.methods)
    method = results.methods{n};
    
    %See if this method has the score we want
    if isfield(scores.(method), testName)
        %If so, record it
        method_scores = [scores.(method).(testName)];
        % Concatenate data by adding a new last dimension
        results.data{n} = cat(3, method_scores.data);
        % Record metadata
        results.name    = union(results.name, {method_scores.name});
        results.metric  = union(results.metric, {method_scores.metric});
        results.optimal = union(results.optimal, [method_scores.optimal]);
        results.range   = union(results.range, [method_scores.range]);
        %Move on to the next method
        n = n + 1;
    else
        %If score not present, remove the method
        results.methods = results.methods([1:(n-1) (n+1):end]);
    end
    
end

% Check that things were consistent
if numel(results.name) ~= 1
    results.name
    error('Inconsistent/missing names!')
else
    results.name = results.name{1};
end
if numel(results.metric) ~= 1
    results.metric
    error('Inconsistent/missing metrics!')
else
    results.metric = results.metric{1};
end
if numel(results.optimal) ~= 1
    results.optimal
    error('Inconsistent/missing optimums!')
end
if numel(results.range) ~= 2
    results.name
    error('Inconsistent/missing ranges!')
else
    results.range = sort(results.range);
end

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [  ] = plotDistributions( methods, results, optimal, lims, visibility )
%Given a set of methods and results, shows the marginal distributions

%Find max number of data points recorded for any method
mLength = 0;
for n = 1:numel(results)
    mLength = max(mLength, numel(results{n}));
end

%Turn the results into a matrix for the boxplot function
boxmat = NaN(mLength, numel(methods));
for n = 1:numel(methods)
    boxmat(1:numel(results{n}), n) = results{n}(:);
end

%Plot results with appropriate labels
figure('Visible', visibility); box on; hold on;
plot([0.5, numel(methods) + 0.5], [optimal, optimal], 'Color', [0.2, 0.8, 0.2]);
%h = boxplot(boxmat, 'labels', methods);%, 'labelorientation', 'inline');
%Change line thicknesses
%for ih=1:6
%    set(h(ih,:),'LineWidth',2);
%end
h = violinplot(boxmat, methods, ...
        'ShowData', false, 'ViolinAlpha', 0.75, 'BoxColor', [0.1, 0.1, 0.1]);
xtickangle(45);

%Set lims
xlim([0.25, numel(methods) + 0.75]); ylim(lims + 0.025 * diff(lims) * [-1; 1]);

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ H ] = makeScatterPlots( methods1, results1, methods2, results2, range )
%Draws scatter plots where the two sets of methods overlap

%Find common methods
[methods, inds1, inds2] = intersect(methods1, methods2);

%Loop over methods
H = [ ];
for n = 1:numel(methods)
    %If they have the same number of results do a scatter plot
    if numel(results1{inds1(n)}) == numel(results2{inds2(n)})
        
        h = figure; H = [H h]; hold on
        %If range has been specified, plot a line
        if nargin == 5
            %This just shows linear trend through the specified ranges
            plot(range.x, range.y, '--', 'Color', 0.8*[1 1 1])
        end
        %Plot the data
        plot(results1{inds1(n)}(:), results2{inds2(n)}(:), ...
            'b.', 'MarkerSize', 10);
        %Similarly set axes lims to the range
        if nargin == 5
            xlim(range.x); ylim(range.y);
        end
        title(methods{n}, 'interpreter', 'none')
        
    end
end

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ ] = plotAllVals( methods, results, optimal, lims, visibility )
%Plots all the results, rather than combining into a single box plot

methodSpacing = 1/2; repeatSpacing = 1/4;

figure('Visible', visibility); hold on; box on

xEnd = 0; labelTicks = NaN(numel(methods,1));
for n = 1:numel(methods)
    
    %Record where this method starts
    xStart = xEnd;
    x = xStart + methodSpacing;
    %Plot dividing line between methods
    if n ~= 1
        plot(xStart*[1 1], lims, '--', 'Color', 0.8*[1 1 1])
    end
    
    %Loop over methods, plotting all their points
    for r = 1:size(results{n},3);
        
        data = results{n}(:,:,r);
        data = data(~isnan(data));
        % Don't plot too much!
        if numel(data) > 500
            data = randsample(data(:), 500);
        end
        % Jitter points in x
        x_data = x + 0.5 * repeatSpacing * (rand([numel(data), 1]) - 0.5);
        % And plot!
        plot(x_data, data(:), 'b.', 'MarkerSize', 5)
        x = x + repeatSpacing; %Advance to next result
        
    end
    
    %Record where we ended up, and where the label should go
    xEnd = (x - repeatSpacing) + methodSpacing;
    labelTicks(n) = (xStart + xEnd) / 2;
    
end

% Plot optimal
plot([0, xEnd], [optimal, optimal], 'Color', [0.2, 0.8, 0.2]);

%Set lims
xlim([0 xEnd]); ylim(lims + 0.025 * diff(lims) * [-1; 1]);

%Add labels
xticks(labelTicks);
xticklabels(methods);
xtickangle(45);

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ H ] = makeAccuracyRetestPlots( PTRmethods, PTRresults, Pmethods, Presults )
%Draws scatter plots of the ground truth accuracy v. the test retest scores

%Find common methods
[methods, indsPTR, indsP] = intersect(PTRmethods, Pmethods);

%Loop over methods
H = [ ];
for n = 1:numel(methods)
    
    %If they have the same number of results do the scatter plot
    if numel(PTRresults{indsPTR(n)}) == numel(Presults{indsP(n)})
        
        h = figure; H = [H h]; hold on
        %Plot a line indicating equality
        plot([0 1], [0 1], '--', 'Color', 0.8*[1 1 1])
        %Also plot a curve showing how scores would be related if the inferred
        %maps were just the truth with additive, independent noise
        x = linspace(0,1,250);
        y = sqrt(x);
        plot(x, y, 'r--')
        
        %Now sort the results based on the test-retest scores
        %This is to help match the accuracy scores (there are two for every
        %split-half score)
        PTR = PTRresults{indsPTR(n)}(:); P = Presults{indsP(n)}(:);
        [PTR,i] = sort(PTR); PTR = PTR(1:2:end);
        %Plot the average accuracy for each split-half score
        P = P(i); P = (P(1:2:end)+P(2:2:end))/2;
        plot(PTR, P, 'r.', 'MarkerSize', 10);
        
        %And now plot all the data
        plot(PTRresults{indsPTR(n)}(:), Presults{indsP(n)}(:), ...
            'b.', 'MarkerSize', 10);
        
        %Set axes lims
        xlim([0 1]); ylim([0 1]);
        axis square
        %Labels
        xlabel('Test-retest reliability')
        ylabel('Ground truth accuracy')
        title(methods{n}, 'interpreter', 'none')
        
    end
end

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
