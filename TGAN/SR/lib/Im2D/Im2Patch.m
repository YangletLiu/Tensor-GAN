function [Y] = Im2Patch(X,par)
% extract patches from a 2D image, and the set of patches are represented
% by a matrix;
% INPUT
%     X : 2D image;
%    par: structure
%         par.patsize: size of blocks
% OUTPUT
%     Y : (patsize x patsize) x N; matrix of blocks;
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
Y           = zeros(patsize*patsize,TotalPatNum);

k=0;
for i = 1:patsize
    for j = 1:patsize
        k= k + 1;   
        tempPatch = X(i:step:end-patsize+i,j:step:end-patsize+j);
        Y(k,:)  = reshape(tempPatch,[1 TotalPatNum]);
    end
end