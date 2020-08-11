function [D] = loadNIfTIs(directory, params)

% SPDX-License-Identifier: Apache-2.0

% Loop over subjects, loading NIfTIs
D = cell(params.S, 1);
for s = 1:params.S
    D{s} = cell(params.R(s), 1);
    subj = sprintf('S%02d',s);
    
    for r = 1:params.R(1)
        run = sprintf('R%02d',r);

        fileName = fullfile(directory, [subj '_' run '.nii.gz']);
        d = read_avw(fileName);
        D{s}{r} = reshape(d, [params.V, params.T]);
    end
end

end
