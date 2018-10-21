function X2 = TwoDSC_R(X,maskP,D0,lambda)
%% hyperparameters settings
dict_size = size(D0,2);
[parDL] = ParSet(X,lambda,dict_size); % set denoising parameters and tensor recovery parameters
%% blocks extraction
% X = V2Block3D(Nmsi,parDL); % Nblock: (patsize x patsize) x N x patsize;
Xhat = fft(X,[],3);
X2 = X;
%% main loop
for iter = 1 : parDL.deNoisingIter
    %===== image blocks recovery by 2DSC=====;
    %% updating B
    if iter == 1
        [B]  = TenTSTA(X2,parDL,D0);
    else
        [B]  = TenTSTA(X2,parDL,D0,B0);
    end
    %% updating D
    [D0] = TenDL(Xhat,B,parDL.r);
    
    %% recovery Y
    [B0] = TenTSTA(X2,parDL,D0);
    
    Y = tensor_prod(D0,[],B0,[]);
    X2 =  Y;
    X2(maskP==1) =  X(maskP==1);
    
    Xhat = fft(X2,[],3);
%     parDL.beta = 0.01 - iter*0.005;   %对参数进行调整（不加入这一步恢复效果不好）

    disp(['Iter: ' num2str(iter),]);
end

end

%===========================================subfunction======================================
function [parDL] = ParSet(X,lambda,dict_size)
% parameters setting for tensor dictioanry learning
parDL.patsize =  size(X, 3);     % size of block;
parDL.Pstep    = 2 ;   % overlapping = patsize - step;
parDL.r       = dict_size;    % number of bases;
parDL.eta     = 1.01 ; % control the increasing speed of Lipschitz constant 
parDL.maxiter  = 10;   % dictionary learning iterations
parDL.maxiterB  = 50;
parDL.deNoisingIter = 5;
parDL.beta = lambda;
end
