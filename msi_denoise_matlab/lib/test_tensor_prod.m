% % test the time for tensor_production
a= rand(25,30,5);
b = rand(25,903224,5);

% cpu
disp('====== cpu =====');
tic
c = tensor_prod(a,'t',b,[]);
t = toc
clear c

% gpu
% gpuDevice(1)
% disp('===== gpu =====');
% tic 
% agpu = gpuArray(a);
% bgpu = gpuArray(b);
% c = tensor_prod_gpu(agpu,'t',bgpu,[]);
% t = toc
% clear c 
% 
% % parts for computing 
disp('===== cpu + part ======');
for partsize = 10:-2:1
    disp(['===== cpu + part ' num2str(partsize) ' ======']);
    tic
    partnum = ceil(size(b,2)/partsize);
    c = zeros(30,903224,5);
    for k = 1:partsize
        idx = (k-1)*partnum + 1:min(k*partnum,size(b,2));
        c(:,idx,:) = tensor_prod(a,'t',b(:,idx,:),[]);
    end
    t = toc
    clear c
end


% fft
% a=rand(25,903224,5);

% cpu fft
% tic
% a=rand(25,903224,5);
% ahat = fft(a,[],3);
% t=toc
% 
% % gpu fft
% tic 
% a=gpuArray(rand(25,903224,5));
% ahat = fft(a,[],3);
% t = toc

