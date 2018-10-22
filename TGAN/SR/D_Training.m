clear all; clc; close all;
% addpath(genpath('RegularizedSC'));
% TR_IMG_PATH = 'Data/Training';
tic;
dataname = 'f3_p';
dataRoad = ['Data/Training/' dataname];
dict_size   = 128;          % dictionary size
lambda      = 0.05;         % sparsity regularization
patch_size  = 4;            % image patch size
nSmp        = 10000;       % number of patches to sample
upscale     = 2;            % upscaling factor 
iter        = 5;
% 
% [Xh, Xl] = rnd_smp_patch(dataRoad, patch_size, nSmp, upscale);
%filename = 'samples\balloons_101_101_31.mat';
%load(filename);
%fg = Omsi(1:100,1:100,:);

%[Xh, Xl] = sample_patches2(fg, patch_size, nSmp, upscale);
[Xh, Xl] = rnd_smp_patch('..\data\mnist_28_28_7.mat', patch_size, nSmp, upscale);
%save('samples\balloons_XhXl.mat', 'Xh', 'Xl')
%load X2wno_diff_double5.mat
%% prune patches with small variances, threshould chosen based on the training data
%  [Xh, Xl] = patch_pruning(Xh, Xl, 10);%要求Xh每列方差大于10,去除平滑样本? 
%% joint sparse coding 
[Dh, Dl] = train_coupled_dict(Xh, Xl, dict_size, lambda, iter);
%dict_path = ['Dictionary/Dque_' num2str(dict_size) '_' num2str(lambda) '_' num2str(patch_size) '_' num2str(iter) '.mat' ];
save('..\data\mnist_28_28_7_dict.mat', 'Dh', 'Dl')
toc