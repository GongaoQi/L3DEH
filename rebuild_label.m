function [F,err_history]=rebuild_label(origin_data,distance_s,param,tol)
%% 1.初始化
rng('default');
rng(param.seed);
Xtrain=origin_data.Xtrain;%text模态
Ytrain=origin_data.Ytrain;%figure模态
L=origin_data.Ltrain;%训练集的模态
Q_x=distance_s.Xtrain;%距离相似度text模态
Q_y=distance_s.Ytrain;
alpha=param.alpha;
eta=param.eta;
beta=param.beta;
delta=param.delta;
lambda=param.lambda;
omega=param.omega;
MaxIter=param.MaxIter;
[X_d,N]=size(Xtrain);
[Y_d,~]=size(Ytrain);
[L_c,~]=size(L);
F=L./sum(L);%初始化重构标签
f1=F;%中间变量
f2=F;
f3=F;
U_x=randn(X_d,L_c);
U_y=randn(Y_d,L_c);
J_x=randn(X_d,L_c);
J_y=randn(Y_d,L_c);
Theta_x=rand(X_d,L_c);%拉格朗日乘子
Theta_y=rand(Y_d,L_c);
err_history = zeros(MaxIter, 1);  % 记录误差

%% 2.迭代
for I=1:MaxIter
 
    F_old=F;

    %% 2.1.更新U
    U_x=(2*Xtrain*F'+2*delta*J_x-Theta_x)/(2*F*F'+2*delta*eye(L_c));
    U_y=(2*Ytrain*F'+2*delta*J_y-Theta_y)/(2*F*F'+2*delta*eye(L_c));
    %% 2.2.更新辅助变量J
    [J_x,~]=SVT_Optimization(U_x+(Theta_x./delta),beta/delta,1,1e-4);
    [J_y,~]=SVT_Optimization(U_y+(Theta_y./delta),beta/delta,1,1e-4);

    %% 2.3更新拉格朗日乘子Theta
    Theta_x=Theta_x+delta*(U_x-J_x);
    Theta_y=Theta_y+delta*(U_y-J_y);

    %% 2.4更新重构标签F
    sumU_x=sum(U_x);
    sumU_y=sum(U_y);
    for maxiter=1:1
        f_old=F;
        sumF_col = sum(F, 1);  

for j = 1:N            
    for i = 1:L_c      
        if L(i,j) ~= 0

            current_col_sum = sumF_col(j) - F(i,j);  

     
            f1_val = (sumU_x(i) + sumU_y(i) - (alpha/2)*(Q_x(i,j)+Q_y(i,j)) + ...
                      omega + lambda - lambda * current_col_sum) / ...
                     (1 + omega + eta + lambda);

            new_val = max(0, min(1, f1_val));
            sumF_col(j) = sumF_col(j) + (new_val - F(i,j));
            F(i,j) = new_val;
        else
            F(i,j) = 0;
        end
    end
end
    end

end

end
