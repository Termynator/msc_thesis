function [ PA ] = makePA( P, A, params )
%Returns PA given the maps and timecourses

% SPDX-License-Identifier: Apache-2.0

%Form the signal as subject maps * scan timecourses
PA = cell(params.S,1);
for s = 1:params.S
    PA{s} = cell(params.R(s),1);
    for r = 1:params.R(s)
        PA{s}{r} = P{s} * A{s}{r};
    end
end

end
