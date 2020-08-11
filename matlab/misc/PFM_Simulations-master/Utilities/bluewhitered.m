function [cmap] = bluewhitered(n)

% SPDX-License-Identifier: Apache-2.0

% Number of colours
if nargin < 1 || isempty(n)
    n = 100;
end

blue  = [0.3, 0.3, 0.8];
white = [1.0, 1.0, 1.0];
red   = [0.9, 0.2, 0.2];

cmap = interp1(...
    [-1.0; 0.0; 1.0], ...
    [blue; white; red], ...
    linspace(-1.0, 1.0, n));

end
