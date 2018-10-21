function [xh] = lin_scale( X, mNorm )
xh = reshape(X,size(X, 1)*size(X, 2),size(X, 3));
hNorm = sqrt(sum(xh(:).^2));
if ~(isreal(hNorm))
    a=1;
end

if real(hNorm),
    s = mNorm*1.2/hNorm;
    xh = xh.*s;
end
xh = reshape(xh,size(X));