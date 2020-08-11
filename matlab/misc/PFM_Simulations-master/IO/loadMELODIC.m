function [ Pg ] = loadMELODIC( melodicDir, params )

% SPDX-License-Identifier: Apache-2.0

Pg = read_avw(fullfile(melodicDir, 'melodic_IC.nii.gz'));
Pg = reshape(Pg, params.V, params.iN);

end
