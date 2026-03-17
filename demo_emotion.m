
clear;

warning('off');

addpath(genpath('metrics'));
addpath(genpath('synthetic_r'));

% load data
load('emotions_1.mat');



data = double(data);
target = double(target);
partial_labels = double(partial_labels);
target = target';
partial_labels = partial_labels';
target(target==-1)=0;
partial_labels(partial_labels==-1)=0;





% Parameter 1: number of anchors (tunable)ÃªµãÊıÁ¿
[~,l] = size(partial_labels);
opt.m=5*l;
numanchor=opt.m;
opt.rho = 1.01;
opt.alpha = 1000;

opt.gamma = 0.1;
opt.beta = 100 ;
opt.knn_k = 5;
opt.max_iter = 50;
opt.mu1 = 1e-1;
opt.mu2=1e-8;
opt.mu3=1e-8;

N = length(target);
indices = crossvalind('Kfold', 1:N ,5);
    
test_idxs = (indices == 1);
train_idxs = ~test_idxs;
        
train_data=data(train_idxs,:);train_target=partial_labels(train_idxs,:);true_target = target(train_idxs,:);
test_data=data(test_idxs,:);test_target=target(test_idxs,:);

% pre-processing
[train_data, settings]=mapminmax(train_data');
test_data=mapminmax('apply',test_data',settings);
train_data(find(isnan(train_data)))=0;
test_data(find(isnan(test_data)))=0;
train_data=train_data';
test_data=test_data';
% 
Y = train_target;
 [~,num_label]=size(train_target);  
%   rand('twister',5489);
[~, A] = litekmeans(Y,numanchor,'MaxIter', 100,'Replicates',10);


% training
model = train_PML(A,train_data, train_target, true_target, opt,test_data,test_target);


clear all;


