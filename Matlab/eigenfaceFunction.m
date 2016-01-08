function [ finalEigenVectors , coVarianceMatrix ] = eigenfaceFunction(trainingImages , d)

%find the mean of the training images
meanTrainingImages = mean(trainingImages ,2);

%meanTRainingImages will be 1x640 matrix
coVarianceMatrix = trainingImages - repmat(meanTrainingImages,[1 size(trainingImages , 2)]);

%find the eigenvector
[U,Diag] = eig(coVarianceMatrix'*coVarianceMatrix);

%sort it by the eigenvalues from greatest to least
[~, idx] = sort(diag(Diag),'descend');
U = U(:,idx);

V = coVarianceMatrix*U;

%now normalize V
for i=1:size(V,2);
    V(:,i) = V(:,i)/sqrt(sum(V(:,i).^2));
end



finalEigenVectors = V(: , 1:d);

end

