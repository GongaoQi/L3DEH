function [B,err_history,TxtToImg,ImgToTxt,sim_t2i,sim_i2t]=retrieval_part_test(RBF_data,r,param_r,rebuild_label_similar,tol)
%% 初始化
rng('default');
rng(param_r.seed);
X=RBF_data.Xtrain;%文本训练集
Y=RBF_data.Ytrain;%图片训练集
L=RBF_data.Ltrain;%原始标签
X_test=RBF_data.Xtest;
Y_test=RBF_data.Ytest;
L_test=RBF_data.Ltest;%测试集标签
lambda=param_r.lambda;
eta=param_r.eta;
%r=param_r.r;%B的位次
alpha=param_r.alpha;
beta=param_r.beta;
xi=param_r.xi;
gama=param_r.gama;
MaxIter=param_r.MaxIter;
S=r.*rebuild_label_similar;%n*n
[X_d,N]=size(X);
[Y_d,~]=size(Y);
[~,N_test]=size(X_test);
B=sgn(randn(r,N));%初始化B
U_x=randn(X_d,r);
U_y=randn(Y_d,r);
V_x=randn(r,N);
V_y=randn(r,N);
% TxtToImg=zeros(MaxIter+1, 1);
% ImgToTxt=zeros(MaxIter+1, 1);
% W_x=B*X'/(X*X'+ xi*eye(X_d));%推广的哈希函数for text,r×X_d
%     W_y=B*Y'/(Y*Y'+ xi*eye(Y_d));%for figure,r×Y_d
%     tBx=sgn(W_x*X_test);%文本检索
%     sim_t2i=B'*tBx;
%     TxtToImg(1) = mAP(sim_t2i,L',L_test',N);
% 
%     tBy=sgn(W_y*Y_test);%图片检索
%     sim_i2t=B'*tBy;
%     ImgToTxt(1) = mAP(sim_i2t,L',L_test',N);




%% 主循环
%Loss = zeros(MaxIter, 1);
%Loss=beta*norm(B-V_x,'fro')^2+beta*norm(B-V_y,'fro')^2+lambda*norm(X-U_x*V_x,'fro')^2+lambda*norm(Y-U_y*V_y,'fro')^2+eta*norm(V_x'*V_y-r*S,'fro')^2+eta*norm(V_y'*V_x-r*S,'fro')^2+alpha*norm(B'*V_x-r*S,'fro')^2+alpha*norm(B'*V_y-r*S,'fro')^2;
err_history1 = zeros(MaxIter, 1);
err_history2 = zeros(MaxIter, 1);
err_history = zeros(MaxIter, 1);
%J_Ux=eye(X_d)+(1/X_d)*(ones(X_d,1)*(ones(X_d,1))');
%J_Uy=eye(Y_d)+(1/Y_d)*(ones(Y_d,1)*(ones(Y_d,1))');
J_Vx=eye(r)+(1/r)*(ones(r,1)*(ones(r,1))');
J_Vy=eye(r)+(1/r)*(ones(r,1)*(ones(r,1))');
% J_Vx=eye(N)+(1/N)*(ones(N,1)*(ones(N,1))');
% J_Vy=eye(N)+(1/N)*(ones(N,1)*(ones(N,1))');
tic;
for I=1:MaxIter
    V_xold=V_x;
    V_yold=V_y;
    B_old=B;
    % Z_x=X*V_x';
    % Z_y=Y*V_y';
    % JUx=J_Ux*Z_x;
    % JUy=J_Uy*Z_y;
    % R_Jx=rank(JUx);
    % R_Jy=rank(JUy);
    % [C_x,~,D_x]=svd(JUx);
    % [C_y,~,D_y]=svd(JUy);
    % C_x1=C_x(:,1:R_Jx);
    % C_y1=C_y(:,1:R_Jy);
    % D_x1=D_x(:,1:R_Jx);
    % D_y1=D_y(:,1:R_Jy);
    % if R_Jx~=r
    % %C_x2=gs_orthogonal([C_x1,ones(X_d,1)],r-R_Jx);
    % C_x2=gs_orthogonal(C_x1,r-R_Jx);%去掉额外加的全1向量，毕竟加全一向量之后，就不是正交矩阵了
    % D_x2=gs_orthogonal(D_x1,r-R_Jx);
    % U_x=(X_d)^(1/2).*([C_x1,C_x2]*[D_x1,D_x2]');
    % else
    %     U_x=(X_d)^(1/2).*(C_x1*D_x1');
    % end
    % if R_Jy~=r
    % %C_y2=gs_orthogonal([C_y1,ones(Y_d,1)],r-R_Jy);
    % C_y2=gs_orthogonal(C_y1,r-R_Jy);%同上
    % D_y2=gs_orthogonal(D_y1,r-R_Jy);
    % U_y=(Y_d)^(1/2).*([C_y1,C_y2]*[D_y1,D_y2]');
    % else
    %     U_y=(Y_d)^(1/2).*(C_y1*D_y1');
    % end

    %% 改变U_x和U_y的迭代方法1

    U_x=lambda*(X*V_x')/(gama+lambda*N);
    U_y=lambda*(Y*V_y')/(gama+lambda*N);
    %U_y=(Y*V_x')/(gama+N);%只有一个U_x和U_y只分解出一个共通的V_x

    
    %% 把U_x和U_y变为映射，且考虑2，1范数

    % M_x=zeros(1,r);
    % for i=1:r
    %     M_x(i)=1/(norm(U_x(i,:),2)+eps);
    % end
    % Mx=0.5*gama*diag(M_x)+eps*eye(r);
    % Mx=inv(Mx);
    % U_x=Mx*((V_x*X')/(X*X'+eye(X_d)));
    % M_y=zeros(1,r);
    % for i=1:r
    %     M_y(i)=1/(norm(U_y(i,:),2)+eps);
    % end
    % My=0.5*gama*diag(M_y)+eps*eye(r);
    % My=inv(My);
    % U_y=My*((V_y*Y')/(Y*Y'+eps*eye(Y_d)));






%% V_x和V_y的迭代方法1
    N_x=(lambda*X'*U_x+r*eta*S*V_y'+r*alpha*S'*B'+beta*B')';
    %N_x=(lambda*X'*U_x+lambda*Y'*U_y+r*alpha*S'*B'+beta*B')';%只有一个U_x和U_y只分解出一个共通的V_x
    %N_x=(lambda*X'*U_x+r*eta*S*V_y'+r*alpha*S'*B')';%去掉V_x-B
    %N_x=(lambda*X'*U_x'+r*eta*S*V_y'+r*alpha*S'*B'+beta*B')';%U_x是映射
    JVx=J_Vx*N_x;
    R_x=rank(JVx);
    [A_x1,~,B_x1]=svds(JVx,R_x);
    %A_x1=A_x(:,1:R_x);
    %B_x1=B_x(:,1:R_x);
    if R_x~=r
        %A_x2=gs_orthogonal([A_x1,ones(r,1)],r-R_x);
        A_x2=gs_orthogonal(A_x1,r-R_x);%同上
        B_x2=gs_orthogonal(B_x1,r-R_x);
        V_x=(N)^(1/2)*([A_x1,A_x2]*[B_x1,B_x2]');
    else
        V_x=(N)^(1/2)*(A_x1*B_x1');
    end

    N_y=(lambda*Y'*U_y+r*eta*S*V_x'+r*alpha*S'*B'+beta*B')';
    %N_y=(lambda*Y'*U_y+r*eta*S*V_x'+r*alpha*S'*B')';%去掉V_y-B
     %N_y=(lambda*Y'*U_y'+r*eta*S*V_x'+r*alpha*S'*B'+beta*B')';%U_y是映射
    JVy=J_Vy*N_y;
    R_y=rank(JVy);
    [A_y1,~,B_y1]=svds(JVy,R_y);
    %A_y1=A_y(:,1:R_y);
    %B_y1=B_y(:,1:R_y);
    if R_y~=r
        %A_y2=gs_orthogonal([A_y1,ones(r,1)],r-R_y);
        A_y2=gs_orthogonal(A_y1,r-R_y);%同上
        B_y2=gs_orthogonal(B_y1,r-R_y);
        V_y=(N)^(1/2)*([A_y1,A_y2]*[B_y1,B_y2]');
    else
        V_y=(N)^(1/2)*(A_y1*B_y1');
    end

%% V_x和V_y的迭代方法_改
    % N_x=(lambda*X'*U_x+r*eta*S*V_y'+r*alpha*S'*B'+beta*B');
    % %N_x=(lambda*X'*U_x+lambda*Y'*U_y+r*alpha*S'*B'+beta*B')';%只有一个U_x和U_y只分解出一个共通的V_x
    % %N_x=(lambda*X'*U_x+r*eta*S*V_y'+r*alpha*S'*B')';%去掉V_x-B
    % %N_x=(lambda*X'*U_x'+r*eta*S*V_y'+r*alpha*S'*B'+beta*B')';%U_x是映射
    % JVx=N_x'*J_Vx*N_x;
    % R_x=rank(JVx);
    % [A_x1,EEX,~]=svds(JVx,R_x);
    % %A_x1=A_x(:,1:R_x);
    % %B_x1=B_x(:,1:R_x);
    % EEX=inv(EEX);
    % B_x1=J_Vx*N_x*A_x1*(EEX)^(1/2);
    % if R_x~=r
    %     %A_x2=gs_orthogonal([A_x1,ones(r,1)],r-R_x);
    %     A_x2=gs_orthogonal(A_x1,r-R_x);%同上
    %     B_x2=gs_orthogonal(B_x1,r-R_x);
    %     V_x=(N)^(1/2)*([A_x1,A_x2]*[B_x1,B_x2]');
    % else
    %     V_x=(N)^(1/2)*(A_x1*B_x1');
    % end
    % 
    % N_y=(lambda*Y'*U_y+r*eta*S*V_x'+r*alpha*S'*B'+beta*B');
    % %N_y=(lambda*Y'*U_y+r*eta*S*V_x'+r*alpha*S'*B')';%去掉V_y-B
    %  %N_y=(lambda*Y'*U_y'+r*eta*S*V_x'+r*alpha*S'*B'+beta*B')';%U_y是映射
    % JVy=N_y'*J_Vy*N_y;
    % R_y=rank(JVy);
    % [A_y1,EEY,~]=svds(JVy,R_y);
    % %A_y1=A_y(:,1:R_y);
    % %B_y1=B_y(:,1:R_y);
    % EEY=inv(EEY);
    % B_y1=J_Vy*N_y*A_y1*(EEY)^(1/2);
    % if R_y~=r
    %     %A_y2=gs_orthogonal([A_y1,ones(r,1)],r-R_y);
    %     A_y2=gs_orthogonal(A_y1,r-R_y);%同上
    %     B_y2=gs_orthogonal(B_y1,r-R_y);
    %     V_y=(N)^(1/2)*([A_y1,A_y2]*[B_y1,B_y2]');
    % else
    %     V_y=(N)^(1/2)*(A_y1*B_y1');
    % end

%% V_x和V_y的迭代方法2

% N_x=(lambda*X'*U_x+r*eta*S*V_y'+r*alpha*S'*B'+beta*B')';
% JVx=N_x'*J_Vx*N_x;
% R_x=rank(JVx);
% [B_x1,Ee1,~]=svds(JVx,R_x);
% %B_x1=B_x(:,1:R_x);
% Eex=sqrtm(Ee1);
% %Eex=Ee_1(1:R_x,1:R_x);
% A_x1=J_Vx*N_x*B_x1/Eex;
% if R_x~=r
% A_x2=gs_orthogonal(A_x1,r-R_x);
% B_x2=gs_orthogonal(B_x1,r-R_x);
% V_x=(N)^(1/2)*([A_x1,A_x2]*[B_x1,B_x2]');
% else
%     V_x=(N)^(1/2)*(A_x1*B_x1');
% end
% 
% N_y=(lambda*Y'*U_y+r*eta*S*V_x'+r*alpha*S'*B'+beta*B')';
% JVy=N_y'*J_Vy*N_y;
% R_y=rank(JVy);
% [B_y1,Ee2,~]=svds(JVy,R_y);
% %B_y1=B_y(:,1:R_y);
% Eey=sqrtm(Ee2);
% %Eey=Ee_2(1:R_y,1:R_y);
% A_y1=J_Vy*N_y*B_y1/Eey;
% if R_y~=r
% A_y2=gs_orthogonal(A_y1,r-R_y);
% B_y2=gs_orthogonal(B_y1,r-R_y);
% V_y=(N)^(1/2)*([A_y1,A_y2]*[B_y1,B_y2]');
% else
%     V_y=(N)^(1/2)*(A_y1*B_y1');
% end





    B=sgn(r*alpha*(V_x*S')+beta*V_x+r*alpha*(V_y*S')+beta*V_y);%更新B
    %B=sgn(r*alpha*(V_x*S')+beta*V_x);%更新B只有一个V_x
    %B=sgn(r*alpha*(V_x*S')+r*alpha*(V_y*S'));%更新B;去掉V_y-B;去掉V_x-B
    %B=sgn(V_x+V_y);


% W_x=B*X'/(X*X'+ xi*eye(X_d));%推广的哈希函数for text,r×X_d
%     W_y=B*Y'/(Y*Y'+ xi*eye(Y_d));%for figure,r×Y_d
%     tBx=sgn(W_x*X_test);%文本检索
%     sim_t2i=B'*tBx;
%     TxtToImg = mAP(sim_t2i,L',L_test',N);
% 
%     tBy=sgn(W_y*Y_test);%图片检索
%     sim_i2t=B'*tBy;
%     ImgToTxt = mAP(sim_i2t,L',L_test',N);
%     fprintf("TxtToImg:%.4f, ImgToTxt:%.4f, A:%.4f\n", TxtToImg, ImgToTxt, ImgToTxt+TxtToImg);

%Loss(I)=beta*norm(B-V_x,'fro')^2+beta*norm(B-V_y,'fro')^2+lambda*norm(X-U_x*V_x,'fro')^2+lambda*norm(Y-U_y*V_y,'fro')^2+eta*norm(V_x'*V_y-r*S,'fro')^2+eta*norm(V_y'*V_x-r*S,'fro')^2+alpha*norm(B'*V_x-r*S,'fro')^2+alpha*norm(B'*V_y-r*S,'fro')^2;

%% 计算误差
    % err_history1(I) = norm(V_x - V_xold, 'fro') / norm(V_xold, 'fro');
    % err_history2(I) = norm(V_y - V_yold, 'fro') / norm(V_yold, 'fro');
    % err_history(I) = norm(B - B_old, 'fro') / norm(B_old, 'fro');
    %     %% 收敛检查
    %     if err_history1(I) < tol && err_history2(I)<tol
    %         fprintf('V_x,V_y收敛于第%d次迭代\n', I);
    %         err_history1 = err_history1(1:I);
    %         err_history2 = err_history2(1:I);
    %         break;
    %     end
end
   train_time = toc;%训练时间;
fprintf('训练耗时: %.2f 秒\n', train_time);
W_x=B*X'/(X*X'+ xi*eye(X_d));%推广的哈希函数for text,r×X_d
W_y=B*Y'/(Y*Y'+ xi*eye(Y_d));%for figure,r×Y_d
    tBx=sgn(W_x*X_test);%文本检索
    sim_t2i=B'*tBx;
    TxtToImg = mAP(sim_t2i,L',L_test',N);
    
    tBy=sgn(W_y*Y_test);%图片检索
    sim_i2t=B'*tBy;
    ImgToTxt = mAP(sim_i2t,L',L_test',N);
    fprintf("TxtToImg:%.4f, ImgToTxt:%.4f, A:%.4f\n", TxtToImg, ImgToTxt, ImgToTxt+TxtToImg);



if I == MaxIter
        fprintf('达到最大迭代次数%d\n', MaxIter);
end
%% 绘制误差曲线
% aaa=[1 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000];
% precision_t2i=zeros(21,1);
% precision_i2t=zeros(21,1);
% for bbb=1:21
% [precision_t2i(bbb),~]=p_r(sim_t2i,L',L_test',aaa(bbb));
% [precision_i2t(bbb),~]=p_r(sim_i2t,L',L_test',aaa(bbb));
% end
end