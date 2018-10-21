function [mMeanP1,mNormP1] = MeanNormExtract(X)

Xdim = permute(X,[1 3 2]);
Xdim = reshape(Xdim,size(Xdim, 1)*size(Xdim, 2),size(Xdim, 3));
mMean = mean(Xdim);
mMeanP1= repmat(mMean, size(Xdim, 1), 1);
Xdim = Xdim - mMeanP1;
mNorm = sqrt(sum(Xdim.^2));%每列平方和开方
mNormP1= repmat(mNorm, size(Xdim, 1), 1);