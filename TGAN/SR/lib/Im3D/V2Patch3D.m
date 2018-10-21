
function  [Y]  =  V2Patch3D( Video, par)
% get full band patches from Video, the third dimension of Video is k;
% INPUT
%      Video: 3D tensor;
%       par : structure
%           par.patsize: size of cubic blocks
%           par.Pstep  : stride
% OUTPUT
%         Y : (patsize x patsize) x N x k;
%-------------------------------------------------------------------
% written by FeiJiang @ sjtu
%-------------------------------------------------------------------
patsize     = par.patsize;
if isfield(par,'Pstep')
    step   = par.Pstep;
else
    step   = 1;
end
TotalPatNum = (floor((size(Video,1)-patsize)/step)+1)*(floor((size(Video,2)-patsize)/step)+1);                  %Total Patch Number in the image
Y           =   zeros(patsize*patsize, TotalPatNum,size(Video,3));                                       %Patches in the original noisy image
count       =   0;

k = size(Video,3);
for i  = 1:patsize
    for j  = 1:patsize
        count     =  count+1;
        tempPatch     =  Video(i:step:end-patsize+i,j:step:end-patsize+j,:);
        Y(count,:,:)      =  reshape(tempPatch,[],k);
    end
end         %Estimated Local Noise Level