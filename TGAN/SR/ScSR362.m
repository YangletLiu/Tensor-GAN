function [hSeis] = ScSR362(seismic_l_o, up_scale, Dh, Dl, lambda, overlap)
%6个导数，整体恢复

%% hyperparameters settings
addpath(genpath('lib'));
[parDL] = ParSet(Dh,overlap,lambda);

%% normalize the dictionary
Dl = normalized(Dl,2);

% patch_size = size(Dh, 3);%处理的图像块大小
nrow = size(seismic_l_o, 1)* up_scale;
ncol = size(seismic_l_o, 2)* up_scale;
nFrames = size(seismic_l_o, 3);

%% generate low resolution counter parts
seismic_l = zeros(nrow, ncol, nFrames);
for k = 1 : nFrames
    im = seismic_l_o(:,:,k);
    lIm = imresize(im,  [nrow ncol], 'bicubic'); %放大up_scale倍
    seismic_l(:,:,k) = lIm;
end
seismic_l(1:up_scale:nrow,1:up_scale:ncol,:) = seismic_l_o;

% extract low-resolution image features
lImfea = extr_lIm_fea(seismic_l);%两个一阶导数、两个二阶导数共4个特征

%% sparse recovery for low-resolution patch
% patch indexes for sparse recovery (avoid boundary)
gridx = 3: nrow-2;
gridy = 3: ncol-2;
%gridz = 3: nFrames-2;
% gridx = 1: nrow;
% gridy = 1: ncol;
gridz = 1: nFrames;
% gridz = 50: 50+patch_size-1;

% features blocks extraction
LP1=[];
for ii = 1:size(lImfea,4)   
    lImG = V2Block3D( lImfea(gridx,gridy,gridz,ii), parDL);      
    LP1= cat(1, LP1, lImG);
end
clear lImG;
LP1 = normalized(LP1,2);

w1  = TenTSTA(LP1,parDL,Dl); 

%% generate the high resolution patch and scale the contrast
HP1 = tensor_prod(Dh,[],w1,[]);  
% blocks extraction
sz_X = size( seismic_l(gridx,gridy,gridz));
[X] = V2Block3D( seismic_l(gridx,gridy,gridz), parDL); 
[mMeanP1,mNormP1] = MeanNormExtract(X);

HP1 = lin_scale_P(HP1, mNormP1, mMeanP1); %去归一化？
%% aggragation
hSeis = zeros(size(seismic_l));%储存处理后的高分辨率数据
hSeis(gridx,gridy,gridz) = Block2V3D(HP1,parDL,sz_X);
idx = (hSeis == 0);
hSeis(idx) = seismic_l(idx);
%%
function [parDL] = ParSet(Dh,overlap,lambda)
% parameters setting for tensor dictioanry learning
parDL.r   =size(Dh, 2);;
parDL.eta     = 1.01 ; % control the increasing speed of Lipschitz constant 
parDL.maxiterB  = 50;
parDL.beta = lambda;
parDL.patsize = size(Dh, 3);%处理的图像块大小
parDL.Pstep = parDL.patsize-overlap;
end

end
% LP = [];
% mNormP = [];
% mMeanP = [];
% 
% for ii = 1:length(gridz),
%     for jj = 1:length(gridy),
%         for kk = 1:length(gridx),
% 
%             zz = gridz(ii);
%             yy = gridy(jj);
%             xx = gridx(kk);
%         
%             mPatch1 = video_l(xx:xx+patch_size-1, yy:yy+patch_size-1,zz:zz+patch_size-1); %取出当前要处理的块
%            
%             mMean = mean(mPatch1(:)); %求块均值
%             mPatch = mPatch1(:)-ones(patch_size*patch_size*patch_size,1)*mMean;
%             mNorm = sqrt(sum(mPatch.^2)); %平方和再开方，2范数？
% 
%             mNormpatch = mNorm * ones(patch_size*patch_size,1,patch_size);
%             mMeanpatch = mMean * ones(patch_size*patch_size,1,patch_size);
% 
%             
%             lImG1 = lImfea(xx:xx+patch_size-1, yy:yy+patch_size-1,zz:zz+patch_size-1,1);
%             lImG2 = lImfea(xx:xx+patch_size-1, yy:yy+patch_size-1,zz:zz+patch_size-1,2);
%             lImG3 = lImfea(xx:xx+patch_size-1, yy:yy+patch_size-1,zz:zz+patch_size-1,3);
%             lImG4 = lImfea(xx:xx+patch_size-1, yy:yy+patch_size-1,zz:zz+patch_size-1,4);
%             lImG5 = lImfea(xx:xx+patch_size-1, yy:yy+patch_size-1,zz:zz+patch_size-1,5);
%             lImG6 = lImfea(xx:xx+patch_size-1, yy:yy+patch_size-1,zz:zz+patch_size-1,6);
%             lImG1 = reshape(lImG1,size(lImG1, 1)*size(lImG1, 2),1,size(lImG1, 3));
%             lImG2 = reshape(lImG2,size(lImG2, 1)*size(lImG2, 2),1,size(lImG2, 3));
%             lImG3 = reshape(lImG3,size(lImG3, 1)*size(lImG3, 2),1,size(lImG3, 3));
%             lImG4 = reshape(lImG4,size(lImG4, 1)*size(lImG4, 2),1,size(lImG4, 3));
%             lImG5 = reshape(lImG5,size(lImG1, 1)*size(lImG1, 2),1,size(lImG1, 3)); 
%             lImG6 = reshape(lImG6,size(lImG1, 1)*size(lImG1, 2),1,size(lImG1, 3));  
%             Lpatch = [lImG1; lImG2; lImG3; lImG4; lImG5; lImG6];
%             Lpatch = normalizedLpatch(Lpatch,2);
%             
%             LP= cat(2, LP, Lpatch);
% 
%             mNormP = cat(2, mNormP, mNormpatch);
%             mMeanP = cat(2, mMeanP, mMeanpatch);
%             
%         end
%     end
% end

% w  = TenTSTA(LP,parDL,Dl);
% % generate the high resolution patch and scale the contrast
% HP = tensor_prod(Dh,[],w,[]);      
% HP = lin_scale_P(HP, mNormP, mMeanP); %去归一化？

% cntMat = zeros(size(seismic_l));%储存，每个位置元素被计算了几次
% i=1;
% for ii = 1:length(gridz)
%     for jj = 1:length(gridy)
%         for kk = 1:length(gridx)
%             
%             zz = gridz(ii);
%             yy = gridy(jj);
%             xx = gridx(kk);
%             
%             hPatch = HP(:,i,:);
%             i = i+1;
%             hPatch = reshape(hPatch,patch_size,patch_size,patch_size);        
%             hSeis(xx:xx+patch_size-1, yy:yy+patch_size-1,zz:zz+patch_size-1) = hSeis(xx:xx+patch_size-1, yy:yy+patch_size-1,zz:zz+patch_size-1) + hPatch;%处理后的高分辨率数据           
%             cntMat(xx:xx+patch_size-1, yy:yy+patch_size-1,zz:zz+patch_size-1) = cntMat(xx:xx+patch_size-1, yy:yy+patch_size-1,zz:zz+patch_size-1) + 1;%计此位置元素计算了几次
%     
%         end
%     end
% end

% fill in the empty with bicubic interpolation
% idx = (cntMat < 1);
% hSeis(idx) = seismic_l(idx);
% cntMat(idx) = 1;
% hSeis = hSeis./cntMat; %根据元素计算次数求平均


