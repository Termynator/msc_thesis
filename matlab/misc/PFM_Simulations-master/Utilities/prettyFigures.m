function [ ] = prettyFigures( )
%PRETTYFIGURES Sets some figure defaults

% SPDX-License-Identifier: Apache-2.0

% get(groot())
% get(gca())

% Text
set(groot(), 'DefaultAxesFontSize', 16);
set(groot(), 'DefaultTextFontSize', 20);
% Turn off underscore -> subscript behaviour
set(groot(), 'DefaultTextInterpreter', 'none')
set(groot(), 'DefaultLegendInterpreter', 'none')
set(groot(), 'DefaultAxesTickLabelInterpreter', 'none')

% Plots
set(groot(), 'DefaultAxesLinewidth', 2);
set(groot(), 'DefaultLineLinewidth', 2);

end
