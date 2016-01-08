[im, labels] = readFaceImages('jaffe');


[imageHeight , imageWidth] = size(im{1});

vectorizedImages = zeros(imageHeight * imageWidth ,211);
imageColumn = 1;
for i= 1:211
        vectorizedImages( : ,imageColumn) = reshape(im{i} , [imageHeight*imageWidth,1]);
        imageColumn = imageColumn + 1;      
end
process = csvread('results.csv');
process = double(process);
perm = randperm(211);

vectorizedImages = vectorizedImages(:,perm(1:210));
labels = labels(perm(1:210));
process = process(perm(1:210),:);
names = {'train1.csv','train2.csv','train3.csv','train4.csv','train5.csv','train6.csv','train7.csv','train8.csv','train9.csv','train10.csv'};
for i=1:10
    start = 1+(i-1)*21;
    ending = 21*i;
    training = horzcat(vectorizedImages(:,1:(start-1)),vectorizedImages(:,(ending+1):end));
    testing = vectorizedImages(:,start:ending);
    new_labels = horzcat(labels(start:ending),labels(1:(start-1)));
    new_labels = horzcat(new_labels,labels(ending+1:end));
    new_proc = vertcat(process(start:ending,:),process(1:(start-1),:));
    new_proc = vertcat(new_proc,process(ending+1:end,:));
    test_labels = labels(start:ending);
    train_labels = horzcat(labels(1:(start-1)),labels(ending+1:end));
    FisherZip(imageHeight , imageWidth, training,testing, train_labels, test_labels, names{i}, new_proc);
end

%EigenfaceRecognition(imageHeight , imageWidth, vectorizedImages,zeros(2500,0), labels, names{1});