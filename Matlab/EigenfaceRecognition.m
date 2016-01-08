function [  ] = EigenfaceRecognition(imageHeight , imageWidth, trainingImages, testingImages, labels, filename)

%Step 1

    [finalEigenVectors , ~] = eigenfaceFunction(trainingImages,10);
    
    % so we have the top 9 eigenfaces
    for i=1:9
         img = reshape(finalEigenVectors(:,i),[imageHeight , imageWidth]);
         figure(1) , subplot(3,3,i);imagesc(img);axis image , axis off , colormap gray 
    end
    
    %eigenvectors has 2500 rows and 9 columns -> each column is an image
   
    %Step 2
    %Represent all face images in the dataset as linear combinations of eigenfaces
    
    X_pca = finalEigenVectors'*(horzcat(testingImages,trainingImages));
    
   % display(X_pca);
    X_pca = X_pca';
    
    for i = 1:210
        dlmwrite(filename,[X_pca(i,:) labels(i)],'delimiter',',','-append');
    end
%     trainingPeople = person(trainingIndices);
% 
%     for i = 1:5
%         testingIndices = subset==i;
%         testingImages = finalEigenVectors' * vectorizedImages(: , testingIndices);
%         testingPeople = person(testingIndices);
%         distanceMatrix = pdist2(X_pca' ,  testingImages');
%         [~ , argmin] = min(distanceMatrix);
%         truth_people = trainingPeople(argmin);
%         display(1 - (nnz(truth_people == testingPeople)/numel(testingPeople)));
%     end
%     
%     %display one original and the corresponding reconstructed face from
%     %each subset.
%     
%     meanTrain = mean(trainingImages , 2);
%     
%     subsets = [6 13 26 39 55]; 
%     sampleImages = vectorizedImages(: , subsets);
%     
%     for i = 1:size(subsets ,2);
%         img = reshape(sampleImages(: , i),[imageHeight , imageWidth]);
%         newImage = finalEigenVectors' * (sampleImages(: , i) - repmat(meanTrain , [1,1]));
%         imrec = meanTrain + finalEigenVectors * newImage;
%         imrec = reshape(imrec,[imageHeight , imageWidth]);
%         figure(2) ,subplot(2,5,i);imagesc(imrec);axis image , axis off , colormap gray
%         figure(2) ,subplot(2,5,i+5);imagesc(img);axis image , axis off , colormap gray
%     end
     
     

    
    
end

