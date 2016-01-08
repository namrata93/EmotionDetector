function [  ] = EigenfaceZip(imageHeight , imageWidth, trainingImages, testingImages, labels, filename, process)

%Step 1

    [finalEigenVectors , ~] = eigenfaceFunction(trainingImages,10);
    
    % so we have the top 9 eigenfaces
    for i=1:9
         img = reshape(finalEigenVectors(:,i),[imageHeight , imageWidth]);
         figure(1) , subplot(3,3,i);imagesc(img);axis image , axis off , colormap gray 
    end
    

    X_pca = finalEigenVectors'*(horzcat(testingImages,trainingImages));
    
   % display(X_pca);
    X_pca = X_pca';
    
    for i = 1:210
        dlmwrite(filename,[process(i,:) X_pca(i,:) labels(i)],'delimiter',',','-append');
    end


    
    
end

