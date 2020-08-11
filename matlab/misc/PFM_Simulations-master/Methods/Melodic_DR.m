function [sicaPg_new,sica_P1_DR_new,sica_A1_DR_new,sica_P1_DR_thres_median,sica_P1_DR_thres_null1,sica_P1_DR_thres_null2,sica_P1_DR_thres_intersect,sica_P1_DR_thres_null3,sica_A1_DR_thres_median,sica_A1_DR_thres_intersect,sica_A1_DR_thres_null1,sica_A1_DR_thres_null2,sica_A1_DR_thres_null3] = Melodic_DR(filename,D,atlasParams,params)

% SPDX-License-Identifier: Apache-2.0

addpath ~steve/NETWORKS/FSLNets;
addpath ~/scratch/matlab/;

% Load existing data and melodic group map
sicaPg_new = read_avw(sprintf('Results/Melodic_%s.gica/melodic_IC.nii.gz',filename));
sicaPg_new = reshape(sicaPg_new,atlasParams.V,params.iN);

% Run DR
sica_P1_DR_new = cell(params.S,1);
sica_A1_DR_new = cell(params.S,1); 
pinv_sicaPg_new = pinv(nets_demean(sicaPg_new)); 
for s = 1:params.S
    sica_A1_DR_new{s} = cell(params.R(s),1);
    M = zeros(atlasParams.V,params.iN,params.R(1));
    for r = 1:params.R(1)
        sica_A1_DR_new{s}{r} = nets_demean((pinv_sicaPg_new * nets_demean(double(D{s}{r})))');
        M(:,:,r) = demean((pinv(demean(double(sica_A1_DR_new{s}{r})))*demean(double(D{s}{r}))')');
    end
    sica_P1_DR_new{s} = mean(M,3);
end

% Run triple regression after thresholding
sica_P1_DR_thres_median = cell(params.S,1);
sica_P1_DR_thres_intersect = cell(params.S,1);
sica_A1_DR_thres_median = cell(params.S,1);
sica_A1_DR_thres_intersect = cell(params.S,1);
sica_A1_DR_thres_null1 = cell(params.S,1);
sica_P1_DR_thres_null1 = cell(params.S,1);
sica_A1_DR_thres_null2 = cell(params.S,1);
sica_P1_DR_thres_null2 = cell(params.S,1);
sica_A1_DR_thres_null3 = cell(params.S,1);
sica_P1_DR_thres_null3 = cell(params.S,1);

for s = 1:params.S
    M = sica_P1_DR_new{s};
        
    % Apply mixture modelling to each map to threshold:
    M1 = zeros(size(M)); M2 = zeros(size(M));
    M3a = zeros(size(M)); M3b = zeros(size(M)); M3c = zeros(size(M));
    for n = 1:size(M,2)
        [stats,thresh] = ggfit(M(:,n));
        M1(M(:,n)<thresh(2),n) = M(M(:,n)<thresh(2),n);
        M1(M(:,n)>thresh(1),n) = M(M(:,n)>thresh(1),n);
        M2(M(:,n)<thresh(4),n) = M(M(:,n)<thresh(4),n);
        M2(M(:,n)>thresh(3),n) = M(M(:,n)>thresh(3),n);
        Mnew = M(:,n)-stats.gaussian.mean;
        Mnew = Mnew/stats.gaussian.std;
        M3a(Mnew<-1,n) = Mnew(Mnew<-1);
        M3a(Mnew>1,n) = Mnew(Mnew>1);
        M3b(Mnew<-2,n) = Mnew(Mnew<-2);
        M3b(Mnew>2,n) = Mnew(Mnew>2);
        M3c(Mnew<-3,n) = Mnew(Mnew<-3);
        M3c(Mnew>3,n) = Mnew(Mnew>3);
    end
    sica_P1_DR_thres_median{s} = M1;
    sica_P1_DR_thres_intersect{s} = M2;
    sica_P1_DR_thres_null1{s} = M3a;
    sica_P1_DR_thres_null2{s} = M3b;
    sica_P1_DR_thres_null3{s} = M3c;
    clear M1 M2 M thres n M3a M3b M3c Mnew
    
%     % Remove overlap (again using mixture modelling to threshold):
%     M = M1;
%     O = sum(abs(M1),2); 
%     [~,thresh] = ggfit(O);
%     M1(O>thresh(3),:) = 0;
%     
%     % If a map is missing entirely, add it back in from the thresholded version:
%     if find(sum(M1)==0); M1(:,sum(M1)==0) = M(:,sum(M1)==0); end
    
    % Obtain timeseries via simple masking:
    sica_A1_DR_thres_intersect{s} = cell(params.R(s),1);
    sica_A1_DR_thres_median{s} = cell(params.R(s),1);
    sica_A1_DR_thres_null1{s} = cell(params.R(s),1);
    sica_A1_DR_thres_null2{s} = cell(params.R(s),1);
    sica_A1_DR_thres_null3{s} = cell(params.R(s),1);
    for r = 1:params.R(1)
        %sica_A1_DR_thres{s}{r} = nets_demean((pinv(nets_demean(M)) * nets_demean(double(D{s}{r})))');
        sica_A1_DR_thres_intersect{s}{r} = nets_demean((pinv(nets_demean(sica_P1_DR_thres_intersect{s})) * nets_demean(double(D{s}{r})))');
        sica_A1_DR_thres_median{s}{r} = nets_demean((pinv(nets_demean(sica_P1_DR_thres_median{s})) * nets_demean(double(D{s}{r})))');
        sica_A1_DR_thres_null1{s}{r} = nets_demean((pinv(nets_demean(sica_P1_DR_thres_null1{s})) * nets_demean(double(D{s}{r})))');
        sica_A1_DR_thres_null2{s}{r} = nets_demean((pinv(nets_demean(sica_P1_DR_thres_null2{s})) * nets_demean(double(D{s}{r})))');
        sica_A1_DR_thres_null3{s}{r} = nets_demean((pinv(nets_demean(sica_P1_DR_thres_null3{s})) * nets_demean(double(D{s}{r})))');
    end
end

% Clear variables
clear s M r 

%%%%%%%%%%%%%% OLD OVERLAP STUFF %%%%%%%%%%%%%%%

%     % normalise and fix sign
%     M1 = M.*repmat(sign(mean(M)),size(M,1),1)./repmat(max(abs(M)),size(M,1),1);
%     [C12,~] = spatialcorr(M1,M); C12 = sign(C12(eye(size(C12,1))==1));
%     M1 = M1.*repmat(C12',size(M1,1),1); 
%     % threshold
%     M1(M1<0.3) = 0;
%     % remove overlap
%     for n = 1:params.N
%         if n == 1; O = M1(:,1); else O = O.*M1(:,n); end
%     end
%     M1(O>0.2,:) = 0;
%     %figure; subplot(1,2,1); imagesc(M,[-40 40]); colorbar; subplot(1,2,2); imagesc(M1,[-1 1]); colorbar  







