function [Y] = Im2Patch3D(X,par)
% extract patches from a 2D image
% INPUT
%     X : 2D image;
%    par: structure
%         par.patsize: size of blocks
% OUTPUT
%     Y : patsize x N x patsize; tensors of blocks;
%----------------------------------------------------
% written by FeiJiang @ sjtu
%----------------------------------------------------

patsize = par.patsize;
if isfield(par,'Pstep')
    step = par.Pstep;
else
    step = 1;
end
TotalPatNum = (floor((size(X,1)-patsize)/step)+1)*(floor((size(X,2)-patsize)/step)+1);
Y           = zeros(patsize,TotalPatNum,patsize);

for i = 1:patsize
    for j = 1:patsize
             
        tempPatch = X(i:step:end-patsize+i,j:step:end-patsize+j);
        Y(i,:,j)  = reshape(tempPatch,[1 TotalPatNum]);
    end
end