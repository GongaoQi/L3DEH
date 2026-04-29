clear all; clc; close all;
warning off;

%% 基本设置
seed = 10;
rng('default');
rng(seed);

%% 参数设置
dataname = 'MIRFLICKR';  % 可选: 'MIRFLICKR', 'IAPRTC-12', 'NUSWIDE', 'Wiki'
param.method = 'L3DEH';

% 重构标签参数
param.alpha = 1e-1;
param.beta = 1e-2;
param.omega = 1e-3;
param.lambda = 5;
param.delta = 1e-2;
param.eta = 1e1;
param.MaxIter = 15;

% 核化参数
param.n_anchors = 1500;

% 检索参数
param.lambda_r = 1e-2;
param.eta_r = 3e3;
param.alpha_r = 1e1;
param.beta_r = 1e1;
param.gama_r = 1e-2;
param.xi_r = 1e-1;
param.MaxIter_r = 2;

% 哈希码位数
bits = [16, 32, 64, 128];

%% 载入数据
fprintf('Loading dataset: %s\n', dataname);
origin_data = load_data(dataname);

%% 计算距离相似度
fprintf('Computing distance similarity...\n');
distance_s.Xtrain = distance_similar(origin_data.Xtrain, origin_data.Ltrain);
distance_s.Ytrain = distance_similar(origin_data.Ytrain, origin_data.Ltrain);

%% 重构标签
fprintf('Rebuilding labels...\n');
[F, err_history_rebuild] = rebuild_label(origin_data, distance_s, param, 0);

%% 计算标签相似度
S = new_similar(F, origin_data.Ltrain);
S(S <= 0) = -1.7;

%% 核化
fprintf('Kernelizing features...\n');
n = size(origin_data.Ltrain, 2);

% 文本模态核化
anchor_text = origin_data.Xtrain(:, randsample(n, param.n_anchors));
RBF_data.Xtrain = RBF_fast(origin_data.Xtrain, anchor_text)';
RBF_data.Xtest = RBF_fast(origin_data.Xtest, anchor_text)';

% 图像模态核化
anchor_image = origin_data.Ytrain(:, randsample(n, param.n_anchors));
RBF_data.Ytrain = RBF_fast(origin_data.Ytrain, anchor_image)';
RBF_data.Ytest = RBF_fast(origin_data.Ytest, anchor_image)';

RBF_data.Ltrain = origin_data.Ltrain;
RBF_data.Ltest = origin_data.Ltest;

%% 检索
fprintf('Starting retrieval...\n');
total_res = [];

for i = 1:length(bits)
    bit = bits(i);
    fprintf('...bit: %d\n', bit);
    
    [B, err_history, TxtToImg, ImgToTxt, sim_t2i, sim_i2t] = retrieval_part_test(origin_data, bit, param, S, 0);
    
    % 保存结果
    total_res = [total_res; ImgToTxt, TxtToImg, bit, ...
                 param.lambda_r, param.eta_r, param.alpha_r, param.beta_r, ...
                 param.gama_r, param.xi_r];
    
    fprintf('  ImgToTxt: %.4f, TxtToImg: %.4f\n', ImgToTxt, TxtToImg);
end

%% 保存结果
save('result.mat', 'total_res');
fprintf('Results saved to result.mat\n');

%% 可选：计算 precision-recall
% F_n = size(origin_data.Ltrain, 2);
% [~, idx] = sort(sim_t2i', 2, 'descend');
% ids_t2i = idx(:, 1:F_n)';
% [precision_t2i, ~] = precision_recall(ids_t2i, origin_data.Ltrain', origin_data.Ltest');
% [~, idx] = sort(sim_i2t', 2, 'descend');
% ids_i2t = idx(:, 1:F_n)';
% [precision_i2t, ~] = precision_recall(ids_i2t, origin_data.Ltrain', origin_data.Ltest');