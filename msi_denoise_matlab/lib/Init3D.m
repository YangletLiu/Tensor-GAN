function [D] = Init3D(par)
% random initialization of tensor dictionary;
% INPUT:
%     par: structure
%         par.patsize: size of blocks
%         par.r      : number of bases;
% OUTPUT
%     D : (patsize x patsize) x r x patsize;
%--------------------------------------------

patsize = par.patsize; % size of blocks
r       = par.r;  % # of bases

Dmat = rand(patsize^3,r)*2-1; % (patsize x patsize x patsize) x r
Dmat = bsxfun(@rdivide,Dmat,sqrt(sum(Dmat.*Dmat,1)));
D    = permute(reshape(Dmat,[patsize^2 patsize r]),[1 3 2]);
end
