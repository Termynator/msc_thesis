function [ r ] = z2r( z )
% Calculates inverse Fisher transformation
% Turns z-scores into corrcoefs
% https://en.wikipedia.org/wiki/Fisher_transformation

% SPDX-License-Identifier: Apache-2.0

r = tanh(z);

end
