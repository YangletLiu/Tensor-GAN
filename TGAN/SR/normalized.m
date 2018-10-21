function Y = normalized(X,opt)
if opt == 1
    Xhdim2 = reshape(X,size(X, 1)*size(X, 3),size(X, 2));
    xNorm = sqrt(sum(Xhdim2.^2));%每列平方和开方
    Y = Xhdim2./repmat(xNorm, size(Xhdim2, 1), 1);%元素除以该列的二范数
    Y = reshape(Y,size(X, 1),size(X, 2),size(X, 3));
end
if opt == 2
    Xhdim2 = permute(X,[1 3 2]);
    Xhdim2 = reshape(Xhdim2,size(Xhdim2, 1)*size(Xhdim2, 2),size(Xhdim2, 3));
    xNorm = sqrt(sum(Xhdim2.^2));%每列平方和开方
    idx = (xNorm == 0);
    xNorm(idx) = 1;
    Y = Xhdim2./repmat(xNorm, size(Xhdim2, 1), 1);%元素除以该列的二范数
    Y = reshape(Y,size(X, 1),size(X, 3),size(X, 2));
    Y = permute(Y,[1 3 2]);
end
if opt == 3
    Xhdim2 = permute(X,[1 3 2]);
    Xhdim2 = reshape(Xhdim2,size(Xhdim2, 1)*size(Xhdim2, 2),size(Xhdim2, 3));
    xNorm = sqrt(sum((abs(Xhdim2)).^2));
    idx = (xNorm == 0);
    xNorm(idx) = 1;
    Y = Xhdim2./repmat(xNorm, size(Xhdim2, 1), 1);%元素除以该列的二范数
    Y = reshape(Y,size(X, 1),size(X, 3),size(X, 2));
    Y = permute(Y,[1 3 2]);
end