function [C] = tensor_prod_gpu(A,ch1,B,ch2)
% computing tensor-production in frequency domain; return C = A * B;
% if ch1 = 't' && ch2 ~= 't', C = A'*B; 
% if ch1 ~= 't' && ch2 == 't', C = A * B;;
% if ch1 == 't' && ch2 == 't', C = A'*B'
%-------------------------------------------------------------------
% by Fei Jiang @ sjtu, 2017/02/21
%-------------------------------------------------------------------
% A = gpuArray(A);
Ahat = fft(A,[],3);
clear A;

% B = gpuArray(B);
Bhat = fft(B,[],3);
clear B;

% step1: size of C
dim = zeros(1,3);
sz_A = size(Ahat); sz_B = size(Bhat);
dim(3) = sz_A(3);
if strcmp(ch1,'t')
    dim(1) = sz_A(2);
else
    dim(1) = sz_A(1);
end
if strcmp(ch2,'t')
    dim(2) = sz_B(1);
else
    dim(2) = sz_B(2);
end

% step2: tensor production in frequency domain
Chat = gpuArray(zeros(dim));


if strcmp(ch1,'t') && strcmp(ch2,'t')
    for k = 1:dim(3)
        Chat(:,:,k) = Ahat(:,:,k)'*Bhat(:,:,k)';
    end
elseif strcmp(ch1,'t')
    for k = 1:dim(3)
       Chat(:,:,k) = Ahat(:,:,k)'*Bhat(:,:,k); 
    end
elseif strcmp(ch2,'t')
    for k = 1:dim(3)
        Chat(:,:,k) = Ahat(:,:,k) * Bhat(:,:,k)';
    end
else
    for k = 1:dim(3)
        Chat(:,:,k) = Ahat(:,:,k) * Bhat(:,:,k);
    end
end
clear Ahat Bhat
% Chat = gather(Chat);
C = ifft(Chat,[],3);
end
    









