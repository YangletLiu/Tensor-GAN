function [Emsi] = TwoDSC_DeNoising_v1(Nmsi,nSig,Omsi)
% MSI denoising based on 2DSC
%
% Input arguments:
%   Nmsi   ...... the MSI corrupted by tensor. Please make sure this MSI is of size height x width x nbands and in range [0, 1].
%   nSig   ...... the variance of noise.
%   Omsi   ...... the clean MSI use to calculate PSNR, this argument is not nesscery.
%
% Output arguments:
%   Emsi   ......  the denoised MSI
%========================================================================================================================
% by FeiJiang @ sjtu
%------------------------------------------------------------------------------------------------------------------------

%% hyperparameters settings
[parDL] = ParSet(nSig); % set denoising parameters and tensor recovery parameters
sz_X = size(Nmsi);
%% blocks extraction
X = V2Block3D(Nmsi,parDL); % Nblock: (patsize x patsize) x N x patsize;
Xhat = fft(X,[],3);

%% initialization about D0
D0 = Init3D(parDL); % randomly initialization of D with size (patsize x patsize) x r x patsize;

%% main loop
for iter = 1 : parDL.deNoisingIter
    %===== image blocks recovery by 2DSC=====;
    %% updating B
    if iter == 1
        [B]  = TenTSTA(X,parDL,D0);
    else
        [B]  = TenTSTA(X,parDL,D0,B0);
    end
    %% updating D
    B(:,1000,:)
    [D0] = TenDL(Xhat,B,parDL.r);
    
    %% recovery Y
    [B0] = TenTSTA(X,parDL,D0);
    Y = tensor_prod(D0,[],B0,[]);

    %% aggragation
    Emsi = Block2V3D(Y,parDL,sz_X);
    
    %% denoising performance
    psnr = PSNR3D(Emsi*255,Omsi*255);
    disp(['Iter: ' num2str(iter),' , current PSNR = ' num2str(psnr)]);
end

end

%===========================================subfunction======================================
function [parDL] = ParSet(nSig)
% parameters setting for tensor dictioanry learning
%

parDL.patsize = 5 ;    % size of block;
parDL.Pstep    = 2 ;   % overlapping = patsize - step;
parDL.r       = 30;    % number of bases;
parDL.eta     = 1.01 ; % control the increasing speed of Lipschitz constant 
parDL.maxiter  = 10;   % dictionary learning iterations
parDL.maxiterB  = 50;
parDL.deNoisingIter = 10;


if nSig <= 0.1
    parDL.beta = 28;
elseif nSig <= 0.2
    parDL.beta = 0.8;
elseif nSig <= 0.3
    parDL.beta = 10;
elseif nSig <= 0.4
    parDL.beta = 290;
end
end



