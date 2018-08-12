function [B,fobj] = TenTSTA(X,par,D0,B0)
% solving for the tensor coefficient B
% min_{S} 0.5 || X - D * B ||_F^2 + beta || B ||_1
%----------------------------------------------------------
% INPUT:
%      X: (patsize x patsize) x N x patsize; % 3D blocks
%    par: structure
%        par.beta: sparsity regularizer
%        par.eta : updating the Lipschitz constant;
%     D0: initialization of the dictionary;
% OUTPUT:
%      B: learned tensor coefficient;
%   fobj: objective value
%----------------------------------------------------------
% by FeiJiang @ sjtu
%----------------------------------------------------------
[~,n,k] = size(X);
r = par.r; % # of atoms;
maxiter = par.maxiterB; % maximum iteration
beta    = par.beta   ; % sparsity regularizer
eta     = par.eta    ; % Lipschitz constant increasing;

if nargin < 3
    D0 = Init3D(par); % randomly initialization;
end
D0tD0 = tensor_prod(D0,'t',D0,[]);
%% Liptisz constant
D0c  = blk_circ_mat(D0tD0);
L0 = norm(D0c);

D0tX = tensor_prod(D0,'t',X,[]);

%% coefficient initialization
if nargin < 4
    B0 = zeros(r,n,k);
end
C1 = B0;
t1 = 1;
fobj = zeros(1,maxiter);
for iter = 1:maxiter
    L1 = eta^iter*L0;                                % lipschitz constant update;
    gradC1 = tensor_prod(D0tD0,'t',C1,[]) - D0tX;    % gradient update
    Temp   = C1 - gradC1./L1;                        % soft-thresholding for L1 constraint
    B1   = sign(Temp).*max(abs(Temp)-beta/L1,0);     % coefficient update
    t2   = (1+sqrt(1+4*t1^2))/2;                     % convergence accelerate
    C1   = B1 + ((t1 - 1)/t2).*(B1 - B0);            % auxilliary tensor update;
    %% update B0 & t1
    B0 = B1;
    t1 = t2;
    %% computing objective value
    fobj(iter) = obj_fun(X,D0,B1,par);
end
B = B1;
return;

% objective function
function [fobj] = obj_fun(X,D,B,par)
% computing the objective value
% obj = 0.5 || X - D*B ||_F^2 + beta ||B||_1;
%--------------------------------------------
beta = par.beta; % sparsity regularizer
diff = X - tensor_prod(D,[],B,[]);

fobj = 0.5*norm(diff(:))^2 + beta*sum(abs(B(:)));
return;



% % randomly intialized the dictionary
% function [D] = Init3D(par)
% % randomly intialization of dictionary
% % INPUT:
% %      par: structure
% %         par.patsize: size of blocks
% %         par.r      : number of atoms;
% % OUTPUT:
% %      D : initialized dictionary;
% %----------------------------------------
%
% patsize = par.patsize;
% r       = par.r;
% dim     = [];