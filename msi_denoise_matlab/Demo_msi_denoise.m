%=================================================================
% demo of 2DSC for msi denoising
%==================================================================
addpath(genpath('lib'));
dataname = 'balloons';
dataRoad = ['msi/data/columbia/' dataname];
load(dataRoad); % load data

saveroad = ['result/result_for_' dataname];
mkdir(saveroad);

rng(1); % fixed the random generator;
%% set enable bits
sigma = 0.2;
disp(['=== The noise level is ' num2str(sigma) ' ===']);

%% resize the msi image 
Smsi = zeros(101,101,31);
for k = 1:size(msi,3)
    Smsi(:,:,k) = imresize(msi(:,:,k),[101 101]);
end

Omsi = normalized(Smsi); % scale the original msi to [0,1]
msi_sz = size(Omsi);
%% add Gaussian noise
noisy_msi = Omsi + sigma * randn(msi_sz); % add Gaussian noise
noisy_msi = Omsi
i = 1;
Re_msi{i} = noisy_msi;
[psnr(i), ssim(i), fsim(i), ergas(i)] = MSIQA(Omsi*255, Re_msi{i}*255); % noisy level 

%% Use TwoDSC_DeNoising method
i = i + 1;

Re_msi{i} = TwoDSC_DeNoising_v1(noisy_msi,sigma,Omsi);

[psnr(i), ssim(i), fsim(i), ergas(i)] = MSIQA(Omsi * 255, Re_msi{i}  * 255);