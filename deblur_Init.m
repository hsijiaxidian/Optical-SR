function net = deblur_Init()
% Create DAGNN object
net = dagnn.DagNN();


% subpixel + conv + relu + conv + bn + relu,  downscale = 2
blockNum = 1;
inVar = 'input';
scale = 1/2;
[net, inVar, blockNum] = addSubP(net, blockNum, inVar, scale);% subp1

dims   = [3,3,4,64];
pad    = [1,1];
stride = [1,1];
lr     = [1,1];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);% conv1
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);% relu3

dims   = [3,3,64,64];
pad    = [1,1];
stride = [1,1];
lr     = [1,0];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
n_ch   = 64;
% [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);% relu5


% conv + bn + relu + conv + bn + relu,  downscale = 4
dims   = [2,2,64,128];
pad    = [0,0];
stride = [2,2];
lr     = [1,0];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
n_ch   = 128;
% [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar); % relu7

dims   = [3,3,128,128];
pad    = [1,1];
stride = [1,1];
lr     = [1,0];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
n_ch   = 128;
% [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);% relu9


% conv + bn + relu + conv + bn + relu,  downscale = 8
dims   = [2,2,128,256];
pad    = [0,0];
stride = [2,2];
lr     = [1,0];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
n_ch   = 256;
% [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar); % relu11

dims   = [3,3,256,256];
pad    = [1,1];
stride = [1,1];
lr     = [1,0];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
n_ch   = 256;
% [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);% relu13


% conv + bn + relu + conv + bn + relu,  downscale = 16
dims   = [2,2,256,512];
pad    = [0,0];
stride = [2,2];
lr     = [1,0];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
n_ch   = 512;
% [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);% relu15

dims   = [3,3,512,512];
pad    = [1,1];
stride = [1,1];
lr     = [1,0];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
n_ch   = 512;
% [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);% relu17


% conv + bnorm + relu
for i = 1 : 1
    dims   = [3,3,512,512];
    pad    = [1,1];
    stride = [1,1];
    lr     = [1,0];
    [net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
    n_ch   = 512;
%     [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch);
    [net, inVar, blockNum] = addReLU(net, blockNum, inVar);% relu19
end


% sum + convt + relu + conv + bnorm + relu
dims = [2,2,256,512];
crop = [0,0];
upsample = 2;
lr   = [1,1];
[net, inVar, blockNum] = addConvt(net, blockNum, inVar, dims, crop, upsample, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
inVar = {'relu13',inVar};
[net, inVar, blockNum] = addSum(net, blockNum, inVar);


dims   = [3,3,256,256];
pad    = [1,1];
stride = [1,1];
lr     = [1,0];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
n_ch   = 256;
% [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);


% sum + convt + relu + conv + bnorm + relu
dims = [2,2,128,256];
crop = [0,0];
upsample = 2;
lr   = [1,1];
[net, inVar, blockNum] = addConvt(net, blockNum, inVar, dims, crop, upsample, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
inVar = {'relu9',inVar};
[net, inVar, blockNum] = addSum(net, blockNum, inVar);


dims   = [3,3,128,128];
pad    = [1,1];
stride = [1,1];
lr     = [1,0];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
n_ch   = 128;
% [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);


% sum + convt + relu + conv + bnorm + relu
dims = [2,2,64,128];
crop = [0,0];
upsample = 2;
lr   = [1,1];
[net, inVar, blockNum] = addConvt(net, blockNum, inVar, dims, crop, upsample, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
inVar = {'relu5',inVar};
[net, inVar, blockNum] = addSum(net, blockNum, inVar);

dims   = [3,3,64,64];
pad    = [1,1];
stride = [1,1];
lr     = [1,0];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
n_ch   = 64;
% [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);


% conv + subp
dims   = [3,3,64,4];
pad    = [1,1];
stride = [1,1];
lr     = [1,1];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
[net, inVar, blockNum] = addSubP(net, blockNum, inVar,2);


inVar = {inVar,'input'};
[net, inVar, blockNum] = addSum(net, blockNum, inVar);

% loss
inVar = {inVar,'label'};
[net] = addLoss(net, blockNum, inVar);

net.vars(net.getVarIndex('sum37')).precious = 1;

%Initialise random parameters


%Visualize Network (340 -> 252)
%net.print({'input', [340 340 1]}, 'all', true, 'format', 'dot')

%Receptive Fields
%net.getVarReceptiveFields('conv36').size

end



function A = clipping(A,b)
A(A>=0&A<b) = b;
A(A<0&A>-b) = -b;
end







% Add a global-std layer
function [net, inVar, blockNum] = addGlobalStd(net, blockNum, inVar)

outVar   = sprintf('globs%d', blockNum);
layerCur = sprintf('globs%d', blockNum);

block = dagnn.GlobalStd();
net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end

% Add a global-pooling layer
function [net, inVar, blockNum] = addGlobalPooling(net, blockNum, inVar)

outVar   = sprintf('globp%d', blockNum);
layerCur = sprintf('globp%d', blockNum);

block = dagnn.GlobalPooling('method','avg');
net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end


% Add a axpy layer
function [net, inVar, blockNum] = addAxpy(net, blockNum, inVar)

outVar   = sprintf('axpy%d', blockNum);
layerCur = sprintf('axpy%d', blockNum);

block = dagnn.Axpy();
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end

% Add a Sigmoid layer
function [net, inVar, blockNum] = addSigmoid(net, blockNum, inVar)

outVar   = sprintf('sigm%d', blockNum);
layerCur = sprintf('sigm%d', blockNum);

block = dagnn.Sigmoid();
net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end

% Add a Tanh layer
function [net, inVar, blockNum] = addTanh(net, blockNum, inVar)

outVar   = sprintf('tanh%d', blockNum);
layerCur = sprintf('tanh%d', blockNum);

block = dagnn.Tanh();
net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end

% Add a Concat layer
function [net, inVar, blockNum] = addConcat(net, blockNum, inVar)

outVar   = sprintf('concat%d', blockNum);
layerCur = sprintf('concat%d', blockNum);

block = dagnn.Concat('dim',3);
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end

% Add a sub-pixel layer
function [net, inVar, blockNum] = addSubP(net, blockNum, inVar, scale)

outVar   = sprintf('subp%d', blockNum);
layerCur = sprintf('subp%d', blockNum);

block = dagnn.SubP('scale',scale);
net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end


% Add a loss layer
function [net, inVar, blockNum] = addLoss(net, blockNum, inVar)

outVar   = 'objective';
layerCur = sprintf('loss%d', blockNum);

block = dagnn.Loss('loss','L2');
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end


% Add a sum layer
function [net, inVar, blockNum] = addSum(net, blockNum, inVar)

outVar   = sprintf('sum%d', blockNum);
layerCur = sprintf('sum%d', blockNum);

block = dagnn.Sum();
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end


% Add a conv layer
function [net, inVar, blockNum] = addReLU(net, blockNum, inVar)

outVar   = sprintf('relu%d', blockNum);
layerCur = sprintf('relu%d', blockNum);

block = dagnn.ReLU('leak',0);
net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end

% Add a conv layer
function [net, inVar, blockNum] = addReLU2(net, blockNum, inVar)

outVar   = sprintf('relu%d', blockNum);
layerCur = sprintf('relu%d', blockNum);

block = dagnn.ReLU('leak',0.2);
net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end




% Add a bnorm layer
function [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch)

b_min = 0.025;
trainMethod = 'Adam';
outVar   = sprintf('bnorm%d', blockNum);
layerCur = sprintf('bnorm%d', blockNum);

params={[layerCur '_g'], [layerCur '_b'], [layerCur '_m']};
net.addLayer(layerCur, dagnn.BatchNorm('numChannels', n_ch), {inVar}, {outVar},params) ;

meanvar  =  [zeros(n_ch,1,'single'), ones(n_ch,1,'single')];


pidx = net.getParamIndex({[layerCur '_g'], [layerCur '_b'], [layerCur '_m']});
%net.params(pidx(1)).value = clipping(sqrt(2/(9*16))*randn(n_ch,1,'single'),b_min);
net.params(pidx(1)).value = max(single(1)+0.02*randn(n_ch,1,'single'),single(0.2));
%net.params(pidx(1)).value = ones(n_ch,1,'single');


net.params(pidx(1)).learningRate= 1;
net.params(pidx(1)).weightDecay = 0;
net.params(pidx(1)).trainMethod = trainMethod;
net.params(pidx(2)).value = zeros(n_ch, 1, 'single');
net.params(pidx(2)).learningRate= 1;
net.params(pidx(2)).weightDecay = 0;
net.params(pidx(2)).trainMethod = trainMethod;
net.params(pidx(3)).value = meanvar;
net.params(pidx(3)).learningRate = 1;
net.params(pidx(3)).weightDecay  = 0;
net.params(pidx(3)).trainMethod  = 'average';




inVar = outVar;
blockNum = blockNum + 1;
end



% add a ConvTranspose layer
function [net, inVar, blockNum] = addConvt(net, blockNum, inVar, dims, crop, upsample, lr)
opts.cudnnWorkspaceLimit = 1024*1024*1024*2; % 2GB
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
trainMethod = 'Adam';

outVar   = sprintf('convt%d', blockNum);

layerCur = sprintf('convt%d', blockNum);

convBlock = dagnn.ConvTranspose('size', dims, 'crop', crop,'upsample', upsample, ...
    'hasBias', true, 'opts', convOpts);

net.addLayer(layerCur, convBlock, {inVar}, {outVar},{[layerCur '_f'], [layerCur '_b']});

f = net.getParamIndex([layerCur '_f']) ;
sc = sqrt(2/(dims(1)*dims(2)*dims(4))) ; %improved Xavier
net.params(f).value = orthrize(randn(dims, 'single')) ;
net.params(f).learningRate = lr(1);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod = trainMethod;

f = net.getParamIndex([layerCur '_b']) ;
net.params(f).value = zeros(dims(3), 1, 'single');
net.params(f).learningRate = lr(2);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar = outVar;
blockNum = blockNum + 1;
end


function A = orthrize(A)
B = A;

A = reshape(A,[size(A,1)*size(A,2)*size(A,3),size(A,4),1,1]);
if size(A,1)> size(A,2)
    [U,S,V] = svd(A,0);
else
    [U,S,V] = svd(A,'econ');
end

S1 =ones(size(diag(S)));
A = U*diag(S1)*V';
A = reshape(A,size(B));
A = single(A);
end

function W = orthrize2(a)

%a = randn(s_(1)*s_(2)*s_(3), s_(4), 'single');
s_ = size(a);
a = reshape(a,[size(a,1)*size(a,2)*size(a,3),size(a,4),1,1]);
[u,d,v] = svd(a, 'econ');
if(size(a,1) < size(a, 2))
    u = v';
end
%W = sqrt(2).*reshape(u, s_);
W = reshape(u, s_);

end


% add a Conv layer
function [net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr)
opts.cudnnWorkspaceLimit = 1024*1024*1024*10; % 2GB
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;

convOpts = {};
trainMethod = 'Adam';

outVar   = sprintf('conv%d', blockNum);

layerCur = sprintf('conv%d', blockNum);

convBlock = dagnn.Conv('size', dims, 'pad', pad,'stride', stride, ...
    'hasBias', true, 'opts', convOpts);

net.addLayer(layerCur, convBlock, {inVar}, {outVar},{[layerCur '_f'], [layerCur '_b']});

f = net.getParamIndex([layerCur '_f']) ;
sc = sqrt(2/(dims(1)*dims(2)*dims(4))) ; %improved Xavier

sc = 1;

net.params(f).value = sc*orthrize2(randn(dims, 'single')) ;
%max(net.params(f).value(:))
%net.params(f).value = sc*(randn(dims, 'single')) ;
net.params(f).learningRate = lr(1);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

f = net.getParamIndex([layerCur '_b']) ;
net.params(f).value = zeros(dims(4), 1, 'single');
net.params(f).learningRate = lr(2);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar = outVar;
blockNum = blockNum + 1;
end


% Add a channel normalization layer
function [net, inVar, blockNum] = addNormalize(net, blockNum, inVar,dims,lr)

trainMethod = 'Adam';
outVar   = sprintf('nml%d', blockNum);
layerCur = sprintf('nml%d', blockNum);

params={[layerCur '_w']};
block = dagnn.Normalize();
net.addLayer(layerCur, block, {inVar}, {outVar}, params);

f = net.getParamIndex([layerCur '_w']);
%net.params(pidx(1)).value = clipping(sqrt(2/(9*16))*randn(n_ch,1,'single'),b_min);
net.params(f(1)).value = orthrize2(randn(dims, 'single'));

net.params(f).learningRate = lr(1);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar = outVar;
blockNum = blockNum + 1;
end