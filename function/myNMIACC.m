function [res]= myNMIACC(U,Y,numclass)
stream = RandStream.getGlobalStream;
reset(stream);
U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1,numclass);
MAXiter = 100; % Maximum number of iterations for KMeans
REPlic = 50; % Number of replications for KMeans
for idx = 1:20
    indx = kmeans(U_normalized,numclass,'maxiter',MAXiter,'replicates',REPlic,'emptyaction','singleton');
    Res(idx,:) = Clustering8Measure(Y, indx);
end
res = [mean(Res); std(Res)];