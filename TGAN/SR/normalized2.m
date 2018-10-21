function [Xh, Xl] = normalized2(Xh,Xl,opt)

if opt == 1
    Xhdim2 = reshape(Xh,size(Xh, 1)*size(Xh, 3),size(Xh, 2));
    Xldim2 = reshape(Xl,size(Xl, 1)*size(Xl, 3),size(Xl, 2));
    xhNorm = sqrt(sum(Xhdim2.^2));
    xlNorm = sqrt(sum(Xldim2.^2));%每列平方和开方
    Idx = find( xhNorm & xlNorm );%返回矩阵非零元素所在位置
    Xhdim2 = Xhdim2(:, Idx);
    Xldim2 = Xldim2(:, Idx);
    Xhdim2 = Xhdim2./repmat(sqrt(sum(Xhdim2.^2)), size(Xhdim2, 1), 1);%元素除以该列的二范数
    Xldim2 = Xldim2./repmat(sqrt(sum(Xldim2.^2)), size(Xldim2, 1), 1);%元素除以该列的二范数
    Xh = reshape(Xhdim2,size(Xh, 1),size(Xh, 2),size(Xh, 3));
    Xl = reshape(Xldim2,size(Xl, 1),size(Xl, 2),size(Xl, 3));
end
if opt==2
    Xhdim2 = permute(Xh,[1 3 2]);
    Xhdim2 = reshape(Xhdim2,size(Xhdim2, 1)*size(Xhdim2, 2),size(Xhdim2, 3));
    Xldim2 = permute(Xl,[1 3 2]);
    Xldim2 = reshape(Xldim2,size(Xldim2, 1)*size(Xldim2, 2),size(Xldim2, 3));
    xhNorm = sqrt(sum(Xhdim2.^2));
    xlNorm = sqrt(sum(Xldim2.^2));
    Idx = find( xhNorm & xlNorm );%返回矩阵非零元素所在位置
    Xhdim2 = Xhdim2(:, Idx);
    Xldim2 = Xldim2(:, Idx);
    Xhdim2 = Xhdim2./repmat(sqrt(sum(Xhdim2.^2)), size(Xhdim2, 1), 1);%元素除以该列的二范数
    Xldim2 = Xldim2./repmat(sqrt(sum(Xldim2.^2)), size(Xldim2, 1), 1);%元素除以该列的二范数
    Xhdim2 = reshape(Xhdim2,size(Xh, 1),size(Xh, 3),size(Xhdim2, 2));
    Xh = permute(Xhdim2,[1 3 2]);
    Xldim2 = reshape(Xldim2,size(Xl, 1),size(Xl, 3),size(Xldim2, 2));
    Xl = permute(Xldim2,[1 3 2]);
end
if opt==3
    Xhdim2 = permute(Xh,[1 3 2]);
    Xhdim2 = reshape(Xhdim2,size(Xhdim2, 1)*size(Xhdim2, 2),size(Xhdim2, 3));
    Xldim2 = permute(Xl,[1 3 2]);
    Xldim2 = reshape(Xldim2,size(Xldim2, 1)*size(Xldim2, 2),size(Xldim2, 3));
    xhNorm = sqrt(sum((abs(Xhdim2)).^2));
    xlNorm = sqrt(sum((abs(Xldim2)).^2));%每列平方和开方
    Idx = find( xhNorm & xlNorm );%返回矩阵非零元素所在位置
    Xhdim2 = Xhdim2(:, Idx);
    Xldim2 = Xldim2(:, Idx);
    Xhdim2 = Xhdim2./repmat(sqrt(sum((abs(Xhdim2)).^2)), size(Xhdim2, 1), 1);%元素除以该列的二范数
    Xldim2 = Xldim2./repmat(sqrt(sum((abs(Xldim2)).^2)), size(Xldim2, 1), 1);%元素除以该列的二范数
    Xhdim2 = reshape(Xhdim2,size(Xh, 1),size(Xh, 3),size(Xh, 2));
    Xh = permute(Xhdim2,[1 3 2]);
    Xldim2 = reshape(Xldim2,size(Xl, 1),size(Xl, 3),size(Xl, 2));
    Xl = permute(Xldim2,[1 3 2]);
end