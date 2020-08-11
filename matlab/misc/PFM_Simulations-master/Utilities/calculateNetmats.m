function [ pcA ] = calculateNetmats( A, params )
%CALCULATENETMATS Returns run-specific partial correlation matrices

% SPDX-License-Identifier: Apache-2.0

pcA = cell(params.S, 1);
for s = 1:params.S
    pcA{s} = cell(params.R(s), 1);
    for r = 1:params.R(s)
        cA = cov(A{s}{r}');
        % Tikhonov regularised partials
        vA = diag(cA);
        vA(vA < 1.0e-3 * max(vA)) = 1.0e-3 * max(vA);
        pcA{s}{r} = - corrcov(inv( cA + diag(0.01 * vA) ));
        % Full correlations
        %pcA{s}{r} = corrcov(cA);
    end
end

end
