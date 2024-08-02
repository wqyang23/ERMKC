function [Hstar, mu, obj, Ztemp] = ERMKC(H,k,lambda,beta,N,V,indexx,indexxavg,Favg)

num = N; %the number of samples
numker = V; %m represents the number of kernels
maxIter = 30; %the number of iterations
% H = zeros(num,k,numker);
%G = eye(num,k,numker);
% G = eye(num);
Z = eye(num);
mu = sqrt(ones(V, 1) / V);
E = cell(V,1);
for i = 1:V
    E{i} = ones(indexx(i),indexxavg);
end
opt.disp = 0;

flag = 1;
iter = 0;
while flag
    iter = iter +1;
    
    J = zeros(num,num);
    K = zeros(num,num);
    for i=1:numker
        J = J + mu(i)*H{i}*H{i}'+beta*Favg*E{i}'*H{i}';
        K = K + mu(i)*H{i}*H{i}'+beta*H{i}*E{i}*E{i}'*H{i}';
    end
    tmp = J/(K+lambda*eye(num));
    for ii=1:num
        idx = 1:num;
        idx(ii) = [];
        Z(ii,idx) = EProjSimplex_new(tmp(ii,idx));
    end
    
    for p = 1:V
        temp = H{p}'*Z'*Favg;
        [Uh,Sh,Vh] = svd(temp,'econ');
        E{p} = Uh*Vh';
    end
    
    f = zeros(V,1);
    for p = 1:V
        f(p) = trace(H{i}'*Z*H{i});
    end
    mu = f ./ norm(f);
    
      
    term1 =0;
    for j =1:numker
        term1 = term1+ mu(j)*(norm((H{j}-Z*H{j}),'fro')^2+beta*norm((Z*H{j}*E{j}-Favg),'fro')^2);
    end

    term2 = lambda*norm(Z,'fro')^2;
    obj(iter) = term1+term2;
    
    if (iter>2) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-6 || iter>maxIter)
       Z= (Z+Z')/2;
       Ztemp = Z;
       Z=Z-diag(Z);
       [Hstar,~] = eigs(Z, k, 'la', opt);
       flag =0;
    end
end