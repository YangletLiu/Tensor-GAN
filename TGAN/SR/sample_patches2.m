function [HP, LP] = sample_patches2(video_h, patch_size, patch_num, upscale)
%采样函数+6个导数
%% 得到低分辨率图像和高分辨率图像
[nrow,ncol,nhei] = size(video_h);
video_l = zeros(nrow,ncol,nhei);
for k = 1 : nhei
    im = video_h(:,:,k);%读取第几帧
    % generate low resolution counter parts
    lIm = im(1:upscale:nrow,1:upscale:ncol);%图像尺寸缩减一半
    lIm = imresize(lIm, size(im), 'bicubic');%图像变回原来的尺寸（低分辨率图像-缩减upscale）
    video_l(:,:,k) =  lIm;
end
% video_l(1:upscale:nrow,1:upscale:ncol,:) = video_h(1:upscale:nrow,1:upscale:ncol,:);
%% queshi
% ratio=0.85;   %保留数据的百分比
% mask=genmask(reshape(video_h,nrow,ncol*nhei),ratio,'r',201415);  %可以调整为列缺失、行缺失和随机缺失
% mask=reshape(mask,nrow,ncol,nhei);
% seismic_l = video_h.*mask;
% 
% idx = (seismic_l == 0);
% seismic_l(idx) = video_l(idx);
% video_l = seismic_l;
%% 低分辨率特征提取
lImfea = extr_lIm_fea(video_l);
clear video_l;

%% 随机选取块左上角元素坐标
x = randperm(nrow-2*patch_size-1) + patch_size;%随机打乱数字序列，再加上块大小
y = randperm(ncol-2*patch_size-1) + patch_size;
%z = randperm(nhei-2*patch_size-1) + patch_size;
z = randperm(nhei-patch_size+1);

%% 取样本块
HP = [];%存储高分辨率图像块
LP = [];%存储低分辨率图像块
ii = 1;
num = 1;
while (ii <= patch_num)
    num = num +1;
    nrow =   randsrc(1,1,x);
    col =   randsrc(1,1,y);
    hei =   randsrc(1,1,z);
    
    Hpatch = video_h(nrow:nrow+patch_size-1,col:col+patch_size-1,hei:hei+patch_size-1);%高分辨率样本块
    Hpatch = Hpatch-ones(patch_size,patch_size,patch_size)*mean(Hpatch(:));%高分辨率样本块-列向量化并减去均值，存储到大矩阵的列向量中
    Hpatch = reshape(Hpatch,size(Hpatch, 1)*size(Hpatch, 2),1,size(Hpatch, 3));  
    Lpatch=[];
    for jj = 1:size(lImfea,4)  
        Lpatch1 = lImfea(nrow:nrow+patch_size-1,col:col+patch_size-1,hei:hei+patch_size-1,jj);
        Lpatch1 = reshape(Lpatch1,size(Lpatch1, 1)*size(Lpatch1, 2),1,size(Lpatch1, 3));
        Lpatch= cat(1, Lpatch, Lpatch1);
    end 
    ii = ii+1;
    HP= cat(2, HP, Hpatch);
    LP= cat(2, LP, Lpatch);
%     j=0;
%     for i = 1:patch_size
%         if  ~(isequal(Hpatch(:,:,i), zeros(patch_size*patch_size,1))) && ~(isequal(Lpatch1(:,:,i) , zeros(patch_size*patch_size*6,1)))
%             j = j+1;
%         end
%     end
%     if j==patch_size  
%         ii = ii+1;
%         HP= cat(2, HP, Hpatch);
%         LP= cat(2, LP, Lpatch);
%     end

end