function Y = TwoDSC(X,parDL)
% sloving following 2DSC-based tensor recovery problem
% min_{D,B} 0.5||X - D*B||_F^2 + beta ||B||_1
% s.t.      ||D(:,j,:)||_F^2 <= 1, j = 1,2,...,r;
%
% INPUT
%    X   ...... the corrupted tensor. m x n x k;
%  parDL ...... an option structure whose fields are as follows:
%      beta ... sparse regularizer, usually setted in [].
%      eta  ... parameter controls the inccreasing speed of Lipschitz  constant
%    maxiter .. max iteration step number
% OUTPUT
%    Y   ...... the reconstruct tensor
%-----------------------------------------------------------------------------------
% by Fei Jiang @ sjtu 2017/02/22
%------------------------------------------------------------------------------------
maxiter = parDL.maxiter;
%% precomputing
Xhat = fft(X,[],3);

%% initialization about D0
D0 = Init3D(parDL); % randomly initialization of D with size (patsize x patsize) x r x patsize;
%% main loop
for iter = 1:maxiter
    
    %% updating B
    [B]  = TenTSTA(X,parDL,D0);
    %% updating D
    [D0] = TenDL(Xhat,B,parDL.r);
    
    %% recovery Y
    [B] = TenTSTA(X,parDL,D0);
    Y = tensor_prod(D0,[],B,[]);
end

return;