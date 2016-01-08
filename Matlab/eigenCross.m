[im, labels] = readFaceImages('jaffe');


[imageHeight , imageWidth] = size(im{1});

vectorizedImages = zeros(imageHeight * imageWidth ,211);
imageColumn = 1;
for i= 1:211
        vectorizedImages( : ,imageColumn) = reshape(im{i} , [imageHeight*imageWidth,1]);
        imageColumn = imageColumn + 1;      
end

perm = randperm(211);

vectorizedImages = vectorizedImages(:,perm(1:210));
labels = labels(perm(1:210));
names = {'train1.csv','train2.csv','train3.csv','train4.csv','train5.csv','train6.csv','train7.csv','train8.csv','train9.csv','train10.csv'};
for i=1:10
    start = 1+(i-1)*21;
    ending = 21*i;
    training = horzcat(vectorizedImages(:,1:(start-1)),vectorizedImages(:,(ending+1):end));
    testing = vectorizedImages(:,start:ending);
    new_labels = horzcat(labels(start:ending),labels(1:(start-1)));
    new_labels = horzcat(new_labels,labels(ending+1:end));
    EigenfaceRecognition(imageHeight , imageWidth, training,testing, new_labels, names{i});
end

%EigenfaceRecognition(imageHeight , imageWidth, vectorizedImages,zeros(2500,0), labels, names{1});