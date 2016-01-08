function [im, finalLabels] = readFaceImages(imdir)


landmarks = csvread('data.csv');
landmarks = double(landmarks);

files = dir(fullfile(imdir, '*.png'));
for f = 1:numel(files)
  fn = files(f).name;
  display(fn);
  getLabel = fn(4:5);
  %person(f) = str2num(fn(7:8));
  %number(f) = str2num(fn(10:11));
  if strcmp(getLabel , 'AN')
     labels(f) = 1;
  elseif strcmp(getLabel , 'DI')
     labels(f) = 2;
  elseif strcmp(getLabel , 'FE')
     labels(f) = 3;
  elseif strcmp(getLabel , 'HA')
     labels(f) = 4;
  elseif strcmp(getLabel , 'SA')
     labels(f) = 5;
  elseif strcmp(getLabel , 'SU')
     labels(f) = 6;
  elseif strcmp(getLabel , 'NE')
    labels(f) =  7;
  end
  
  temp_im = im2single(imread(fullfile(imdir, fn)));
  min_c = min(landmarks(f,1:66));
  max_c = max(landmarks(f,1:66));
  min_r = min(landmarks(f,67:132));
  max_r = max(landmarks(f,67:132));
  temp_im = temp_im(min_r:max_r,min_c:max_c);
  temp_im = imresize(temp_im , [50 , 50]);
  im{f} = temp_im;
end

finalLabels = labels;





% files = dir(fullfile(imdir, '*.gif'));
% for f = 1:numel(files)
%   fn = files(f).name;
%   id(f) = str2num(fn(8:9));
%   im{f} = imread(fullfile(imdir, fn));
% end
