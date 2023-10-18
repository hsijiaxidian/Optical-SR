
% clear; clc;

%% testing set
%addpath(fullfile('data','utilities'));


imageSets   = {'Set5','Set14','Set68','classic5','LIVE1'};
image_set   = imageSets{5};
%% model information
folderTest = fullfile('R1');

showresult  = 1;

epoch       = 45;

%% load model
modelName   = 'deblur';
load(fullfile('data',modelName,[modelName,'-epoch-',num2str(epoch),'.mat']));
net = dagnn.DagNN.loadobj(net) ;


net.removeLayer('loss62') ;

out1 = net.getVarIndex('sum61') ;
net.vars(net.getVarIndex('sum61')).precious = 1 ;

net.mode = 'test';
gpu = 1;
if gpu
    net.move('gpu');
end

%%% read images
ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
end

% %%% PSNR and SSIM
% PSNRs = zeros(1,length(filePaths));
% SSIMs = zeros(1,length(filePaths));


% sigma = 15/255;
% l = sqrt(sum(1/64*4+1/4));
% s0 = 1;
% s1 = 2;
% s2 = 3;
%
% sigma = 35/255;
% l = sqrt(sum(1/64*4+1/4));
% s0 = 1;
% s1 = 1;
% s2 = 1;



% l0 = 10/255
% l1 = 10/255
% l2 = 40/255


for i = 1 : length(filePaths)
    %%% read images
    label = imread(fullfile(folderTest,filePaths(i).name));
    
    label = modcrop(label,8);
    %     if size(label,3)==3
    %         label = rgb2ycbcr(label);
    %     end
    
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    kernel   = generate_motion_blur;
    kernel = fspecial('gaussian',64,4);
    
    %kernel1 = fspecial('gaussian',64,1.25);
    %     kernel(1:63,1:63) = kernel1;
    
    blur_HR  = imfilter(label,double(kernel),'replicate');
    blur_HR  = imfilter(label,double(kernel),'circular', 'conv');
    
    
    blur_B = blur_HR;
    H = size(blur_B,1);    W = size(blur_B,2);
    blur_B_w = wrap_boundary_liu(blur_B, opt_fft_size([H W]+size(kernel)-1));
    blur_B_tmp = blur_B_w(1:H,1:W,:);
    
    blur_HR = uint8(blur_B_tmp);
    
    LR1      = im2single(im2uint8(blur_HR));
    inputs = im2single(LR1);
    N = 1;
    inputs = inputs + N/255*randn(size(inputs));
    %kernel = fspecial('gaussian',64,0.1);
    
    k = kernel;
    sigma = N/255;
    y = inputs;
    [w,h,c]  = size(y);
    V = psf2otf(k,[w,h]);
    
    %imshow(real(V))
    
    denominator = abs(V).^2;
    z = single(y);
    if c>1
        denominator = repmat(denominator,[1,1,c]);
        V = repmat(V,[1,1,c]);
    end
    upperleft   = conj(V).*fft2(y);
    lamda = (sigma^2)/3;
    rho = lamda*255^2/(40^2) ;
    z = real(ifft2((upperleft + rho*fft2(z))./(denominator + rho)));
    
  %  kernel = fspecial('gaussian',64,0.1);
    
    map1 = bsxfun(@times,ones(size(inputs,1)/8,size(inputs,2)/8,1,size(inputs,4)),permute(single(kernel(:)),[3 4 1 2]));
    map1 = vl_nnSubP(map1,[],'scale',8);
    map2  = bsxfun(@times,ones(size(inputs,1),size(inputs,2),1,size(inputs,4)),permute(single([N/255;0/255]),[3 4 1 2]));
    map = cat(3,map1,map2);
    tic;
    inputs  = gpuArray(inputs);
    map    = gpuArray(map);
    
    %     if gpu
    %         input = gpuArray(input);
    %     end
    net.eval({'input', inputs, 'map', map}) ;
    %%% output (single)
    output = gather(squeeze(gather(net.vars(out1).value)));
    %     output = gather(squeeze(gather(net.vars(out2).value))) + input_bic;
    toc;
    kernel = imresize(kernel,1,'nearest')/max(kernel(:));
    kernel = cat(3,kernel,kernel,kernel);
    label(1:size(kernel,1),1:size(kernel,2),:) = im2uint8(kernel);
    
    %%% calculate PSNR and SSIM
    % [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),scaleP(1),scaleP(1));
    if showresult
        % imshow(im2uint8(output));
        %title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        imshow(cat(2,im2uint8(gather(inputs)),im2uint8(output),im2uint8(z),label));
        
        drawnow;
        % pause()
    end
    %     PSNRs(i) = PSNRCur;
    %     SSIMs(i) = SSIMCur;
end




% disp([mean(PSNRs),mean(SSIMs)]);



