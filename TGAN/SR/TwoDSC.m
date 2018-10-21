function [D0] = TwoDSC(X,dict_size,lambda, iter)

%% hyperparameters settings
[parDL] = ParSet(X,lambda,dict_size, iter); % set denoising parameters and tensor recovery parameters
% sz_X = size(X);
%% blocks extraction
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
    [D0] = TenDL(Xhat,B,parDL.r);
    
    %% recovery Y
    [B0] = TenTSTA(X,parDL,D0);

    %% denoising performance
    disp(['Iter: ' num2str(iter)]);
    
end

end

%===========================================subfunction======================================
function [parDL] = ParSet(X,lambda,dict_size, iter)
% parameters setting for tensor dictioanry learning
parDL.patsize =  size(X, 3);     % size of block;
parDL.Pstep    = 2 ;   % overlapping = patsize - step;
parDL.r       = dict_size;    % number of bases;
parDL.eta     = 1.01 ; % control the increasing speed of Lipschitz constant 
parDL.maxiter  = 10;   % dictionary learning iterations
parDL.maxiterB  = 50;
parDL.deNoisingIter = iter;
parDL.beta = lambda;
end



