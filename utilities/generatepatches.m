function [imdb] = generatepatches


folder     = 'D:\Myworks\2018\Optical_SR\Trainingdata\SEM_input_train';
folder2     = 'D:\Myworks\2018\Optical_SR\Trainingdata\SEM_output_train';


scale      = 1;

stride     = 25;
size_label = 96;
batchSize  = 64;

count      = 0;
numscales  = 1;

stride_low = stride/scale;
step1      = randi(stride_low-10)*scale;
step2      = randi(stride_low-10)*scale;

%scalesc    = min(1,0.5 + 0.05*randi(15));


size_input = size_label;
padding    = abs(size_input - size_label)/2;
% scales     = [1:0.5:4];

ext               =  {'*.jpg','*.png','*.bmp','*.tif'};
filepaths         =  [];
for i = 1 : length(ext)
    filepaths = cat(1,filepaths, dir(fullfile(folder, ext{i})));
end

rdis = randperm(length(filepaths));
ntar = round(length(filepaths));


% scalesc    = min(1,0.5 + 0.05*randi(15,[1,ntar]));
nn         = randi(8,[1,ntar]);

for i = 1 : ntar
    im = imread(fullfile(folder,filepaths(rdis(i)).name));
    
    im = modcrop(im,8);
    
  %  im = data_augmentation(im, nn(i));
    disp([i,ntar,round(count/256)])
    %   for j= 1:numscales
    %im = imresize(im,scalesc(i),'bicubic');
    %         rng((i-1)*numscales + j);
    %         scalesr = scales(randperm(length(scales)));
    %        scale   = scalesr(j);
    %[im]    = imresizef(im, 1);
    LR_label = ones([size(im,1)/scale,size(im,2)/scale]);
    [hei,wid,~] = size(im);
    for x = 1+step1 : stride : (hei-size_input+1)
        for y = 1+step2 : stride : (wid-size_input+1)
            x_l = stride_low*(x-1)/stride + 1;
            y_l = stride_low*(y-1)/stride + 1;
            if x_l+size_input/scale-1 > size(LR_label,1) || y_l+size_input/scale-1 > size(LR_label,2)
                continue;
            end
            %  subim_input = im_input(x_l : x_l+size_input/scale-1, y_l : y_l+size_input/scale-1);
            %                 subim_lrlabel = LR_label(x_l : x_l+size_input/scale-1, y_l : y_l+size_input/scale-1);
            %                 subim_label = HR_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);
            count=count+1;
            %                 HRlabels(:, :, 1, count) = subim_label;
            %                 LRlabels(:, :, 1, count) = subim_lrlabel;
            % data(:, :, 1, count) = subim_input;
        end
    end
    % end
end


%%------------------------------------------------------------------%%
%%------------------------------------------------------------------%%


numPatches = ceil(count/batchSize)*batchSize;
diffPatches = numPatches - count;
disp([numPatches,numPatches/batchSize,diffPatches]);

disp('---------------PAUSE------------');
%pause


filepaths2         =  [];
for i = 1 : length(ext)
    filepaths2 = cat(1,filepaths2, dir(fullfile(folder2, ext{i})));
end

count = 0;
imdb.LRlabels  = zeros(size_label/scale, size_label/scale, 1, numPatches,'single');
imdb.HRlabels  = zeros(size_label, size_label, 1, numPatches,'single');
% imdb.kernelsum = zeros(kernelsize*kernelsize, numPatches,'single');
% imdb.sigmasum  = zeros(2,numPatches,'single');

for i = 1 : ntar
    LR = imread(fullfile(folder,filepaths(rdis(i)).name));
    LR = modcrop(LR,8);
    %LR = rgb2ycbcr(LR);
    LR = im2single(LR(:,:,1));
   % LR = data_augmentation(LR, nn(i));
    
    HR = imread(fullfile(folder2,filepaths2(rdis(i)).name));
    HR = modcrop(HR,8);
    %HR = rgb2ycbcr(HR);
    HR = im2single(HR(:,:,1));
  %  HR = data_augmentation(HR, nn(i));   
    
    disp([i,ntar,round(count/256)])
    
    for j= 1:numscales % length(scales)
        
%         im = imresize(im,scalesc(i),'bicubic');
%         im = im2double(im);
        %         rng((i-1)*numscales + j);
        %         scalesr = scales(randperm(length(scales)));
        %         scale   = scalesr(j);
        %[~, LR, HR, kernel, Nsigma] = imresizef(im);
        
        [hei,wid,~] = size(HR);
        LR_label = LR;
        HR_label = HR;
        for x = 1+step1 : stride : (hei-size_input+1)
            for y = 1+step2 : stride : (wid-size_input+1)
                x_l = stride_low*(x-1)/stride + 1;
                y_l = stride_low*(y-1)/stride + 1;
                if x_l+size_input/scale-1 > size(LR_label,1) || y_l+size_input/scale-1 > size(LR_label,2)
                    continue;
                end
                %  subim_input = im_input(x_l : x_l+size_input/scale-1, y_l : y_l+size_input/scale-1);
                subim_lrlabel = LR_label(x_l : x_l+size_input/scale-1, y_l : y_l+size_input/scale-1,1:1);
                subim_label   = HR_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,1:1);
                count         = count + 1;
                imdb.HRlabels(:, :, :, count) = subim_label;
                imdb.LRlabels(:, :, :, count) = subim_lrlabel;
%                 imdb.kernelsum(:,count)       = single(kernel(:));
%                 imdb.sigmasum(:,count)        = Nsigma;
                
                
                if count<=diffPatches
                    imdb.LRlabels(:, :, :, end-count+1)   = LR_label(x_l : x_l+size_input/scale-1, y_l : y_l+size_input/scale-1,1:1);
                    imdb.HRlabels(:, :, :, end-count+1)   = HR_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,1:1);
%                     imdb.kernelsum(:,end-count+1)       = single(kernel(:));
%                     imdb.sigmasum(:,end-count+1)          = single(Nsigma);
                end
            end
        end
    end
end

imdb.set    = uint8(ones(1,size(imdb.LRlabels,4)));


