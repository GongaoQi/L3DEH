function origin_data = load_data(dataname, varargin)
% 载入数据
% 可选参数: 'NUSWIDE' 时可以传入样本数量

switch dataname
    case 'MIRFLICKR'
        load("MIRFLICKR.mat");
        origin_data.Xtest = T_te';
        origin_data.Ytest = I_te';
        origin_data.Ltest = L_te';
        origin_data.Xtrain = T_tr';
        origin_data.Ytrain = I_tr';
        origin_data.Ltrain = L_tr';

    case 'IAPRTC-12'
        load("IAPRTC-12.mat");
        origin_data.Xtest = T_te';
        origin_data.Ytest = I_te';
        origin_data.Ltest = L_te';
        origin_data.Xtrain = T_tr';
        origin_data.Ytrain = I_tr';
        origin_data.Ltrain = L_tr';

    case 'NUSWIDE'
        load("NUSWIDE.mat");
        % 默认采样20000个样本
        if nargin > 1
            n_samples = varargin{1};
        else
            n_samples = 20000;
        end
        inx = randperm(size(L_tr,1), n_samples);
        origin_data.Xtest = T_te';
        origin_data.Ytest = I_te';
        origin_data.Ltest = L_te';
        origin_data.Xtrain = T_tr';
        origin_data.Ytrain = I_tr';
        origin_data.Ltrain = L_tr';
        origin_data.Ltrain = origin_data.Ltrain(:, inx);
        origin_data.Xtrain = origin_data.Xtrain(:, inx);
        origin_data.Ytrain = origin_data.Ytrain(:, inx);

    case 'Wiki'
        load("WikiData.mat");
        origin_data.Xtest = T_te';
        origin_data.Ytest = I_te';
        origin_data.Ltest = L_te';
        origin_data.Xtrain = T_tr';
        origin_data.Ytrain = I_tr';
        origin_data.Ltrain = L_tr';

    otherwise
        error('Unknown dataset: %s', dataname);
end
end