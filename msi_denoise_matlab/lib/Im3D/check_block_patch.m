% check the correctness of the aggragation 
a = rand(10,10,10);
par.patsize = 5;
par.step    = 3;

%% extract full band patches
[b] = V2Patch3D(a,par);

% aggragation
[c] = Patch2V3D(b,par,size(a));

% check the equalence
[m1 n1 k1] = size(a);
[m11 n11 k11] = size(c);

m = min(m1,m11);
n = min(n1,n11);
k = min(k1,k11);

a1 = a(1:m,1:n,1:k);
c1 = c(1:m,1:n,1:k);

norm(a1(:)-c1(:))

%% extract blocks
[d] = V2Block3D(a,par);
[e] = Block2V3D(d,par,size(a));

% check the equalence
[m1 n1 k1] = size(a);
[m11 n11 k11] = size(e);

m = min(m1,m11);
n = min(n1,n11);
k = min(k1,k11);

a1 = a(1:m,1:n,1:k);
e1 = e(1:m,1:n,1:k);

norm(a1(:)-e1(:))





