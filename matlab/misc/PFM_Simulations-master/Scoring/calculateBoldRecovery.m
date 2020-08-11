function [ PArms ] = calculateBoldRecovery( PA, infPA, params )
%Returns accuracy of inferred BOLD signal
%   Returns the RMS error between ground truth and inferred spaces

% SPDX-License-Identifier: Apache-2.0

%RMS error between the BOLD signals
PArms = 0;
for s = 1:params.S
    for r = 1:params.R(s)
        PArms = PArms + sum(sum( (PA{s}{r} - infPA{s}{r}).^2 ));
    end
end
PArms = PArms / (sum(params.R) * params.V * params.T);
PArms = sqrt(PArms);

end
