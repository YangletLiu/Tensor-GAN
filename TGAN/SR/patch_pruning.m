function [Xh, Xl] = patch_pruning(Xh, Xl, threshold)
Xhdim2 = reshape(Xh,size(Xh, 1),size(Xh, 2)*size(Xh, 3));
Xldim2 = reshape(Xl,size(Xl, 1),size(Xl, 2)*size(Xl, 3));

pvars = var(Xhdim2, 0, 1);%对每列方差

idx = pvars > threshold;

Xhdim2 = Xhdim2(:, idx);
Xldim2 = Xldim2(:, idx);
Xh = reshape(Xhdim2,size(Xh, 1),size(Xh, 2),size(Xh, 3));
Xl = reshape(Xldim2,size(Xl, 1),size(Xl, 2),size(Xl, 3));