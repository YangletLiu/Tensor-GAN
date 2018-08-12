function [E_V,Weight] = Patch2V3D(VPat,par,sizeV)
% recovery the original 3D image from the estimated patches
% ImPat: 由算法生成的blocks集合； 为三阶张量；sizeV:原始video的size；
%-----------------------------------------------------------------
% INPUT:
%       VPat: (patsize x patsize) x N x k; k is the third dimension of
%              original video
%        par: structure
%            par.patsize: size of blocks
%            par.Pstep  : stride for the nearby blocks
%       sizeV: the original size of the 3D video
% OUTPUT:
%       E_Img: recovery video
%       Weight: the weights of each pixels
%------------------------------------------------------------------
% written by FeiJiang @ sjtu
%-------------------------------------------------------------------

patsize = par.patsize;
if isfield(par,'Pstep')
    step = par.Pstep;
else
    step = 1;
end
TempR    = floor((sizeV(1)-patsize)/step)+1;
TempC    = floor((sizeV(2)-patsize)/step)+1;
TempOffsetR = [1:step:(TempR-1)*step+1];
TempOffsetC = [1:step:(TempC-1)*step+1];
xx = length(TempOffsetR); % Video的row中可以提取的patches数目
yy = length(TempOffsetC); % Video的column中可以提取的patches数目

E_V       = zeros(sizeV);
Weight    = zeros(sizeV);

k = sizeV(3);
N = size(VPat,2);
count = 0;
for i = 1:patsize
    for j = 1:patsize
        count = count + 1;
        E_V(TempOffsetR-1+i,TempOffsetC-1+j,:)    = E_V(TempOffsetR-1+i,TempOffsetC-1+j,:)    + reshape(VPat(count,:,:),[xx yy k]);
        Weight(TempOffsetR-1+i,TempOffsetC-1+j,:) = Weight(TempOffsetR-1+i,TempOffsetC-1+j,:) + ones(xx,yy,k);
    end
end

E_V = E_V ./(Weight+eps);

% removing the zero rows and columns
E_V = E_V(1:(TempR-1)*step+patsize,1:(TempC-1)*step+patsize,:);
