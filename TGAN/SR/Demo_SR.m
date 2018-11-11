clear all; 
clc;
close all;
addpath(genpath('S4M'));
tic;
%% set parameters
lambda = 0.01;                  % sparsity regularization
overlap = 3 ;                    % the more overlap the better (patch size 5x5)
upscale = 2;                   % scaling factor, depending on the trained dictionary
% maxIter = 20;                   % if 0, do not use backprojection
n_pic = 10;
%% read test data and upscale
%load('samples/balloons_101_101_31.mat');
% Testdata = fg(1:106,1:106,1:106);
%Testdata = Omsi(1:100,1:100,:);
load('..\data\mnist_28_28_7.mat');
load('..\data\mnist_28_28_7_dict.mat');
load('..\data\mnist_test_14_14_7.mat');
YY = 255*YY;
ns = randperm(size(YY,1));
for i=1:1:n_pic
    Testdata = squeeze(YY(ns(i),:,:,:));
    [nrow,ncol,nFrames] = size(Testdata);
    seismic_l = Testdata(1:upscale:nrow,1:upscale:ncol,:);%Í¼Ïñ³ß´çËõ¼õÒ»°ë
    seismic_l_half = Testdata(1:upscale:nrow,:,:);%Í¼Ïñ³ß´çËõ¼õÒ»°ë
%% super-resolution
% load dictionary

%load('Dictionary/Dque_128_0.05_5_5.mat');
% super-resolution based on sparse representation
    [seismic_h] = ScSR362(seismic_l, upscale, Dh, Dl, lambda, overlap);
    seismic_h (1:upscale:nrow,1:upscale:ncol,:) = Testdata(1:upscale:nrow,1:upscale:ncol,:);
    seismic_h(isnan(seismic_h)) = 0;
    %seismic_h(find(seismic_h<0))=0;
    subplot(2,n_pic,i);
    image(seismic_l(:,:,4));
    set(gca,'xtick',[],'xticklabel',[])
    set(gca,'ytick',[],'yticklabel',[])
    subplot(2,n_pic,n_pic+i);
    image(seismic_h(:,:,4))
    set(gca,'xtick',[],'xticklabel',[])
    set(gca,'ytick',[],'yticklabel',[])

end
RSE = norm(seismic_h(:) - Testdata(:)) / norm(Testdata(:));
% backprojection
%  for k = 3
%      [video_h_y(:,:,k)] = backprojection(video_h_y(:,:,k), video_l_y(:,:,k), maxIter);         
%  end
%% compute PSNR&RMSE 
sei_o= Testdata(:,:,5);
sei_p= Testdata(:,:,5);
sei_p(2:upscale:nrow,:)=0;
sei_p(:,2:upscale:ncol)=0;
sei_l= seismic_l(:,:,5);

sei_h = seismic_h(:,:,5);
% % bicubic interpolation for reference
sei_b = imresize(sei_l, [nrow, ncol], 'bicubic');

sei_o=sei_o(3:102,3:102);
sei_p=sei_p(3:102,3:102);
sei_h=sei_h(3:102,3:102);
sei_b=sei_b(3:102,3:102);
p_RSE =  norm(sei_p(:) - sei_o(:)) / norm(sei_o(:));
sp_RSE = norm(sei_h(:) - sei_o(:)) / norm(sei_o(:));
bb_RSE = norm(sei_b(:) - sei_o(:)) / norm(sei_o(:));
p_rmse = compute_rmse(sei_o, sei_p);
bb_rmse = compute_rmse(sei_o, sei_b);
sp_rmse = compute_rmse(sei_o, sei_h);

maxdata = max(sei_o(:));
p_psnr = 20*log10(maxdata/p_rmse);
bb_psnr= 20*log10(maxdata/bb_rmse);
sp_psnr = 20*log10(maxdata/sp_rmse);
%fprintf('PSNR for l: %f dB\n', l_psnr);
fprintf('PSNR for Bicubic Interpolation: %f dB\n', bb_psnr);
fprintf('PSNR for Sparse Representation Recovery: %f dB\n', sp_psnr);
%fprintf('RMSE for l: %f dB\n', l_rmse);
fprintf('RMSE for Bicubic Interpolation: %f dB\n', bb_rmse);
fprintf('RMSE for Sparse Representation Recovery: %f dB\n', sp_rmse);
toc 

%% show the images
Residual = sei_h-sei_o;
Residual_Bicubic = sei_b-sei_o;
scale = max(Residual(:)) / max(sei_o(:));
SeisPlot(sei_o, {'figure', 'new'}, {'scale', 1/1500});%title('Original');
SeisPlot(sei_l, {'figure', 'new'}, {'scale', 1/1500}),%title('LD');
%SeisPlot(sei_p, {'figure', 'new'}, {'scale', 1/1500}),%title('LD');
SeisPlot(sei_h,{'figure', 'new'}, {'scale', 1/1000});%title('HD');
SeisPlot(sei_b,{'figure', 'new'}, {'scale', 1/1500});%title('Bicubic Interpolation');
SeisPlot(Residual, {'figure', 'new'}, {'scale', scale/1500});%title('Residual');
SeisPlot(Residual_Bicubic, {'figure', 'new'}, {'scale', scale/1500});%title('Residual_Bicubic');
sei_l_half=seismic_l_half(:,:,5);
SeisPlot(sei_l_half, {'figure', 'new'}, {'scale', 1/1500}),%title('LD');

t=1:100; 
figure,plot(t,sei_o(19,:),'r',t,sei_h(19,:),'--*b',t,sei_h(19,:)-sei_o(19,:),'-o');
set(gca,'fontsize',20)
legend('Original', 'Proposed algorithm', 'Residual');

figure,plot(t,sei_o(19,:),'r',t,sei_b(19,:),'--*b',t,sei_b(19,:)-sei_o(19,:),'-o');
set(gca,'fontsize',20)
legend('Original', 'Bicubic Interpolation', 'Residual');
t=1:30; 
figure,plot(t,sei_o(19,1:30),'r',t,sei_h(19,1:30),'--*b',t,sei_h(19,1:30)-sei_o(19,1:30),'-o');

 