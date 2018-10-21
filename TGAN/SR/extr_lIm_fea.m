function [lImFea] = extr_lIm_fea( seismic_l )

[nrow,ncol,nFrames] = size(seismic_l);

seismic_l_lImG1 = zeros(nrow, ncol, nFrames);
seismic_l_lImG2 = zeros(nrow, ncol, nFrames);
seismic_l_lImG3 = zeros(nrow, ncol, nFrames);
seismic_l_lImG4 = zeros(nrow, ncol, nFrames);

hf1 = [-1,0,1];
vf1 = [-1,0,1]';
hf2 = [1,0,-2,0,1];
vf2 = [1,0,-2,0,1]';
for k = 1 : nFrames
    lIm = seismic_l(:,:,k);%¶ÁÈ¡µÚ¼¸Ö¡
    % generate low resolution counter parts
    seismic_l_lImG1(:,:,k) = conv2(lIm, hf1,'same');
    seismic_l_lImG2(:,:,k) = conv2(lIm, vf1,'same');
    seismic_l_lImG3(:,:,k) = conv2(lIm,hf2,'same');
    seismic_l_lImG4(:,:,k) = conv2(lIm,vf2,'same');
end
seismic_l = permute(seismic_l,[1,3,2]);
seismic_l_lImG5 = zeros(nrow, nFrames, ncol);
seismic_l_lImG6 = zeros(nrow, nFrames, ncol);
for k = 1 : size(seismic_l, 3)
    lIm = seismic_l(:,:,k);
    % compute the first and second order gradients
    seismic_l_lImG5(:,:,k) = conv2(lIm, hf1,'same');
    seismic_l_lImG6(:,:,k) = conv2(lIm, hf2,'same');
end
seismic_l_lImG5 = permute(seismic_l_lImG5,[1,3,2]);
seismic_l_lImG6 = permute(seismic_l_lImG6,[1,3,2]);

lImFea= cat(4, seismic_l_lImG1, seismic_l_lImG2, seismic_l_lImG3, seismic_l_lImG4, seismic_l_lImG5, seismic_l_lImG6);
