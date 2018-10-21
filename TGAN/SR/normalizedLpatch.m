function Y = normalizedLpatch(X,opt)
if opt == 1
    Xdim2 = reshape(X,size(X, 1)*size(X, 2),size(X, 3));
    xNorm = sqrt(sum(Xdim2.^2));%每列平方和开方
    for i = 1:length(xNorm)
        if xNorm(i) > 1
            xNorm(i) = xNorm(i);%特征归一化？
        else
            xNorm(i) =1;
        end
    end
    Y = Xdim2./repmat(xNorm, size(Xdim2, 1), 1);%元素除以该列的二范数
    Y = reshape(Y,size(X));
end
if opt == 2
    Xdim2 = X(:);
    xNorm = sqrt(sum(Xdim2.^2));%每列平方和开方
    if xNorm > 1
        xNorm = xNorm;%特征归一化？
    else
        xNorm =1;
    end
    Y = Xdim2./repmat(xNorm, size(Xdim2, 1), 1);%元素除以该列的二范数
    Y = reshape(Y,size(X));
end