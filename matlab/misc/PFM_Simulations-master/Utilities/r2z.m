function [ z ] = r2z( r )
% Calculates Fisher transformation
% Turns corrcoefs into z-scores
% https://en.wikipedia.org/wiki/Fisher_transformation

% SPDX-License-Identifier: Apache-2.0

z = atanh(r);

end
