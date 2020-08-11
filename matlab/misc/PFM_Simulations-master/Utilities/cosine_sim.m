function [ C ] = cosine_sim( X, Y )
% Calculates cosine similarity (between columns)
% https://en.wikipedia.org/wiki/Cosine_similarity
% Same behaviour as `corr(X)`,`corr(X, Y)`

% SPDX-License-Identifier: Apache-2.0

if nargin < 2
    Y = X;
end

x2 = sum(X.^2); x2(x2 < eps) = 1.0;
y2 = sum(Y.^2); y2(y2 < eps) = 1.0;

C = (X' * Y) ./ (sqrt(x2)' * sqrt(y2));

end
