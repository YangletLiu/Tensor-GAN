
function  [Y]  =  V2Block3D( Video, par)
% get cubic blocks extracted from video; video is  n1 x n2 x n3;
% INPUT
%      Video: 3D tensor;
%       par : structure
%           par.patsize: size of cubic blocks
%           par.Pstep  : stride
% OUTPUT
%         Y : (patsize x patsize) x N x patsize;
%-------------------------------------------------------------------
% written by FeiJiang @ sjtu
%-------------------------------------------------------------------

patsize     = par.patsize;
if isfield(par,'Pstep')
    step   = par.Pstep;
else
    step   = 1;
end
TotalPatNum = (floor((size(Video,1)-patsize)/step)+1)*(floor((size(Video,2)-patsize)/step)+1)*(floor((size(Video,3)-patsize)/step)+1);
%Total Patch Number in the image :floor向下取整
Z           =   zeros(patsize,patsize, patsize, TotalPatNum);         
%Patches in the original noisy image

for i  = 1:patsize
    for j  = 1:patsize
        for k = 1:patsize
            tempPatch     =  Video(i:step:end-patsize+i,j:step:end-patsize+j,k:step:end-patsize+k);
            Z(i,j,k,:)    =  reshape(tempPatch,[1 TotalPatNum]);
        end
    end
end         %Estimated Local Noise Level

Y = permute(reshape(Z,[patsize*patsize,patsize,TotalPatNum]),[1 3 2]);