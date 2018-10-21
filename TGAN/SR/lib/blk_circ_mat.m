function [Ac] = blk_circ_mat(A)
% return the block circulant matrix of tensor A;
% INPUT:
%      A: m x n x k: tensor
% OUTPUT:
%     Ac: mk x nk matrix
%-----------------------------------------------
% by Fei Jiang @ sjtu 2017/02/21
%-----------------------------------------------

sz_A = size(A);
dim = zeros(1,2); 
dim(1) = sz_A(1)*sz_A(3);  dim(2) = sz_A(2)*sz_A(3);

Ac = zeros(dim);
Amat = reshape(permute(A,[1,3,2]),[dim(1),sz_A(2)]);
Ac(:,1:sz_A(2)) = Amat;
for k = 2:sz_A(3)
    Ac(:,(k-1)*sz_A(2)+1:k*sz_A(2)) = circshift(Amat,(k-1)*sz_A(1));    
end