function saveNIfTIs(D, directory, params)

% SPDX-License-Identifier: Apache-2.0

assert(mod(params.V, 100) == 0, '`params.V` must be divisible by 100!')

% Prepare NIfTI
vsize = [1 1 1 params.TR];
vtype = 'f';

% Loop over subjects, saving NIfTIs and creating MELODIC/PROFUMO sped files
data = struct();
fileNames = {};
for s = 1:params.S
    subj = sprintf('S%02d',s);
    
    for r = 1:params.R(1)
        run = sprintf('R%02d',r);
        
        data3d = reshape(D{s}{r}, 10, 10, params.V / 100, params.T);
        fileName = fullfile(directory, [subj '_' run '.nii.gz']);
        save_avw(data3d, fileName, vtype, vsize);
        
        data.(subj).(run) = fileName;
        fileNames{end+1} = fileName;
    end
end

% Save spec files
fileID = fopen(fullfile(directory, 'PROFUMO_SpecFile.json'), 'w');
fprintf(fileID, '%s\n', jsonencode(data));
fclose(fileID);
fileID = fopen(fullfile(directory, 'MELODIC_SpecFile.txt'), 'w');
fprintf(fileID, '%s\n', fileNames{:});
fclose(fileID);
fileID = fopen(fullfile(directory, 'DualRegression_SpecFile.txt'), 'w');
fprintf(fileID, '%s ', fileNames{:});
fclose(fileID);

end
