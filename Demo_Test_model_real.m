
% clear; clc;

%% testing set
addpath(fullfile('utilities'));
%% model information

folderTest = fullfile('D:\Myworks\2018\Optical_SR\Trainingdata\SEM_input_train');
folderTest2 = fullfile('D:\Myworks\2018\Optical_SR\Trainingdata\SEM_output_train');
Result_folder = fullfile('D:\Myworks\2018\Optical_SR\Results\');
showresult  = 1;

epoch       =32;

%% load model
modelName   = 'DnCNN';
load(fullfile('data','DnCNN',[modelName,'-epoch-',num2str(epoch),'.mat']));
net = dagnn.DagNN.loadobj(net) ;

net.removeLayer('loss37') ;

out1 = net.getVarIndex('sum36') ;
net.vars(net.getVarIndex('sum36')).precious = 1 ;

net.mode = 'test';
gpu = 1;
if gpu
    net.move('gpu');
end

%%% read images
ext         =  {'*.jpg','*.png','*.bmp','*.tif'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
end
ext         =  {'*.jpg','*.png','*.bmp','*.tif'};
filePaths2   =  [];
for i = 1 : length(ext)
    filePaths2 = cat(1,filePaths2, dir(fullfile(folderTest2,ext{i})));
end
% %%% PSNR and SSIM
% PSNRs = zeros(1,length(filePaths));
% SSIMs = zeros(1,length(filePaths));


for i = 1: length(filePaths)
    %%% read images
    input = imread(fullfile(folderTest,filePaths(i).name));
%    input = imresize(input,[1870,1870],'bicubic');
%     input = modcrop(input,96,0);
    %iycbcr = rgb2ycbcr(input);
%     input = shave(input,[96,96]);
     input2 = imread(fullfile(folderTest2,filePaths2(i).name));
%    input2 = imresize(input2,[770,770]);
%     input2 = modcrop(input2,16*6,0);
    %iycbcr = rgb2ycbcr(input);
%     input2 = shave(input2,[96,96]);
    
    
    size(input)
    
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    
    inputs = im2single(input);
%     inputs = sign(max(inputs-0.1,0)).*inputs;
    
    
    
    tic;
    inputs  = gpuArray(inputs);
    %  map    = gpuArray(map);
    
    %     if gpu
    %         input = gpuArray(input);
    %     end
    net.eval({'input', inputs}) ;
    %%% output (single)
    output = squeeze(gather(net.vars(out1).value));
    %     output = gather(squeeze(gather(net.vars(out2).value))) + input_bic;
    toc;

    output = im2uint16(output);
    % [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),scaleP(1),scaleP(1));
    
    if showresult
        % imshow(im2uint8(output));
        %title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        imshow(cat(2,im2uint16(gather(input)),im2uint16(output),im2uint16(input2),im2uint16(input2-input)));
        
       imwrite(output,[Result_folder, nameCur,'_CNN482_deblur.tif']);
       imwrite(input,[Result_folder, nameCur,'.tif']);
       imwrite(input2,[Result_folder, nameCur,'_GT.tif']); 
        
        
        
        
        drawnow;
        pause()
        
    end
    
    %     PSNRs(i) = PSNRCur;
    %     SSIMs(i) = SSIMCur;
end








% imwrite(im2uint8(gather(inputs)),'11.png')
% imwrite(im2uint8(output),'12.png')
% disp([mean(PSNRs),mean(SSIMs)]);



