
%%% Note: run the 'GenerateData_model_64_25_Res_Bnorm_Adam.m' to generate
%%% training data first.


%rng('default')

format compact;
%%%-------------------------------------------------------------------------
%%% Configuration
%%%-------------------------------------------------------------------------
addpath('D:\Myworks\2018\Optical_SR\utilities');
opts.modelName        = 'deblur'; %%% model name
opts.learningRate     = [logspace(-4,-4,50) logspace(-4,-5,50) logspace(-5,-7,300)];%%% you can change the learning rate
opts.batchSize        = 64; %%% 
opts.gpus             = [1]; %%% this code can only support one GPU!
opts.numSubBatches    = 2;
opts.bnormLearningRate= 0;

%%% solver
opts.solver           = 'Adam';
opts.numberImdb       = 1;
opts.imdbDir          = 'D:\Myworks\2018\Optical_SR\Trainingdata';
opts.derOutputs       = {'objective',1} ;

opts.gradientClipping = false; %%% set 'true' to prevent exploding gradients in the beginning.
opts.backPropDepth    = Inf;
%%%-------------------------------------------------------------------------
%%%   Initialize model and load data
%%%-------------------------------------------------------------------------
%%%  model

net  = feval([opts.modelName,'_Init']);

%%%  load data
opts.expDir      = fullfile('data', opts.modelName);

%%%-------------------------------------------------------------------------
%%%   Train
%%%-------------------------------------------------------------------------

[net, info] = deblur_train_dag(net,  ...
    'expDir', opts.expDir, ...
    'learningRate',opts.learningRate, ...
    'derOutputs',opts.derOutputs, ...
    'bnormLearningRate',opts.bnormLearningRate, ...
    'numSubBatches',opts.numSubBatches, ...
    'numberImdb',opts.numberImdb, ...
    'backPropDepth',opts.backPropDepth, ...
    'imdbDir',opts.imdbDir, ...
    'solver',opts.solver, ...
    'gradientClipping',opts.gradientClipping, ...
    'batchSize', opts.batchSize, ...
    'modelname', opts.modelName, ...
    'gpus',opts.gpus) ;






