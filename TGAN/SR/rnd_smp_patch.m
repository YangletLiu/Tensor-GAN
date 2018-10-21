function [Xh, Xl] = rnd_smp_patch(dataRoad, patch_size, num_patch, upscale)

load(dataRoad);
sz = size(XX);
ns = randperm(sz(1));
Xh = [];
Xl = [];

for i=1:1:2000
    Xs = squeeze(XX(ns(i),:,:,:));
    [Ph, Pl] = sample_patches2(Xs, patch_size, 5, upscale);
    Xh = cat(2,Xh,Ph);
    Xl = cat(2,Xl,Pl);
end


