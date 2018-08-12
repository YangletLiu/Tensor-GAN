function [B] = l2ls_learn_basis_dual_v2(Xhat,S,num_bases)
% learning the tensor basis using Lagrange dual (with basis normalization)
% INPUT:
%     X: training data: m x n x k;
%     S: coefficient  : r x n x k;
%    Binit: initialization of B: m x r x k;
% OUTPUT:
%    B: m x r x k;

Shat = fft(S,[],3);
clear S;
%% step1: FFT %%%
dual_lambda = 10*abs(rand(num_bases,1));

%% SSt & XSt
[m,~,k] = size(Xhat); r=num_bases;
SSt = gpuArray(zeros(r,r,k)); XSt = gpuArray(zeros(m,r,k));
for kk = 1:k
    xhatk = Xhat(:,:,kk); 
    shatk = Shat(:,:,kk);
    
    SSt(:,:,kk) =shatk*shatk';
    XSt(:,:,kk) = xhatk*shatk';
end

%% dual-Lagrange 
lb = zeros(size(dual_lambda));
options = optimset('Algorithm','trust-region-reflective' ,'GradObj','on', 'Hessian','on','Display','off');

[x] = fmincon(@(x) fobj_basis_dual(x,XSt,SSt,k), dual_lambda,[],[],[],[],lb,[],[],options);

Lambda = gpuArray(diag(x));
Bhat = gpuArray(zeros(m,r,k)); 
for kk = 1:k
    SStk = squeeze(SSt(:,:,kk));
    XStk = squeeze(XSt(:,:,kk));
    Bhatkt = pinv(SStk + Lambda) * XStk';
    Bhat(:,:,kk) = Bhatkt';  
end

B = ifft(Bhat,[],3);
B(find(isnan(B))) = 0;
B = real(B);

% Bmat = reshape(permute(B,[1,3,2]),[m*k,r]);
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f,g,H] = fobj_basis_dual(lambda,XSt,SSt,k)
% computing the dual function and its gradient
% dual function: D(lambda) = \sum_{j=1}^{n}||Xj-BjSj||^2 + \sum_p=1^r
% lambda_p (\sum_j=1^n ||Bj(:,p)||_F^2 - c)
% where Bj = (Xj^TSj)(SjSj^T+Lambda)^-1
lambda = gpuArray(lambda);
m = size(XSt,1); r=length(lambda); Lambda = diag(lambda);

f=gpuArray(0); g = gpuArray(zeros(r,1)); H = gpuArray(zeros(r,r));
for kk = 1:k
   XStk = XSt(:,:,kk);
   SStk = SSt(:,:,kk);
   SSt_inv = pinv(SStk+Lambda);
   
   if m > r
       f = f+trace(SSt_inv*(XStk'*XStk));
   else
       f = f + trace(XStk*SSt_inv*XStk');
   end
   
   Bkt = SSt_inv * XStk';
   if nargout > 1
      g = g -  diag(Bkt*Bkt');
      H = H +2*(Bkt*Bkt').*(SSt_inv);  
      H = gather(real(H));
   end
   
end

f = real(f + k*sum(lambda));
g = real(g + k);

f = gather(f);
g = gather(g);

return;
