function Y = normalized(X)
Xdim2 = reshape(X,size(X, 1)*size(X, 3),size(X, 2));
xNorm = sqrt(sum(Xdim2.^2));%每列平方和开方
Y = Xdim2./repmat(xNorm, size(Xdim2, 1), 1);%元素除以该列的二范数
Y = reshape(Y,size(X));
end