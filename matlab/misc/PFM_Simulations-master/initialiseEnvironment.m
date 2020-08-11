function [ ] = initialiseEnvironment( )
%INTIIALISEENVIRONMENT Sets paths etc

% SPDX-License-Identifier: Apache-2.0

% FSL functions
addpath(fullfile(getenv('FSLDIR'), 'etc', 'matlab'));

% Internal paths
addpath(genpath('Utilities/'));
addpath('DataGeneration/');
addpath('IO/');
addpath('Methods/');
addpath('Scoring/');
addpath('Visualisation/');

end
