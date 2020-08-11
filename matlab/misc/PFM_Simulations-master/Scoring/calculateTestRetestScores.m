function [ Pscore1, Pscore2 ] = calculateTestRetestScores( infP1, infP2, params )
%CALCULATESCORES Returns accuracy of inferred maps and BOLD signal
%   Returns the mean correlation of the 

% SPDX-License-Identifier: Apache-2.0

%Calculate the map scores first
cPtotal = 0;
for s = 1:params.S
    
    %Take cosine similarity between real and observed maps
    cP = [infP1{s} infP2{s}]' * [infP1{s} infP2{s}];
    cP = corrcov(cP);
    %Just the scores between the two
    cP = cP( 1:params.iN, params.iN+(1:params.iN) );
    cPtotal = cPtotal + cP;
    
end
cPtotal = cPtotal / params.S;

%Match with the Hungarian algorithm
matching = Hungarian( -abs( cPtotal ) );
%Return scores in order of inferred components
Pscore2 = abs(cPtotal(matching==1));
cPtotal = cPtotal'; matching = matching';
Pscore1 = abs(cPtotal(matching==1));

end
