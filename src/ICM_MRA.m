function [Dx,vessel] = ICM_MRA(IMG,Iout,K,Object,SelecNum,NB,alfa,gama,IL,MPL_alfa,Beta_ini,iterations)

% 函数意义：只用来分割脑部MRA数据中的血管
% 采用了Iout_vessel_perscent函数估计血管的初始均值、方差、初始阈值threthold
% 与run_ICM_MRA_segmentation的区别：采用MPL法全自动估计高级MRF模型参数
% Iout为血管响应矩阵，由Frangi的多尺度血管增强函数得到
% K为总的分类数，Object为分割目标类号（此函数针对血管类Object=4）
% alfa:多尺度血管滤波响应阈值的加权系数，一般为1.5
% 通常 SelecNum = [1 2 3 4];
% 示例： Dx = ICM_MRA(Img,Iout,4,4,[1 2 3 4],6,1.5,0.01,3,0.3,0);
Iout = double(Iout);
IMG = double(IMG);
close all; pause(0.1);
c = size(IMG,3);
%----------------------------------------- 数据分类：并计算其均值、方差、比例系数
[VBN_EM,criValue,~] = SegParameter_MRA_pdf_curl(IMG,Iout,K,alfa,iterations);%计算IMG数据的分类参数、概率曲线拟合、参数更新；显示figure(1~3)
VBN = VBN_EM;
[flagIMG] = GetRegion(Iout,gama*criValue); % 获取经验分布空间flagIMG,criValue为临界阈值
% flagIMG=ones(size(IMG));
[Dx_MLE,sort_D] = ML_estimation(IMG,VBN,Object,flagIMG);  % 极大似然估计的初始标记场
inPLX = sort_D(:,5);%
pxl_k = sort_D(:,1:4);
figure(4);imshow_Process(IMG,Iout,Dx_MLE);pause(0.5);
% figure(4);subplot(2,1+LSN,2+LSN);imshow3D_patch(flagIMG,flagIMG,[0.5 0.5 0.5]);title('血管分布初始空间');pause(1);% 显示血管的3D初始空间
disp(['候选空间体素数为' num2str(length(find(flagIMG==1))) '； 总体素数为' num2str(numel(flagIMG)) '；候选空间比率为' num2str(100*length(find(flagIMG==1))/numel(flagIMG)) '%']);
% figure(5);[OptiumBeTa] = BeTa_estimation(IMG,VBN,flagIMG,4,Dx_init,NB,MPL_alfa,Beta_ini);
%  OptiumBeTa = 0.3;
OptiumBeTa = (NB==6)*0.7 +(NB==0)*0.7 + (NB==26)*0.19;
disp(['OptiumBeTa = ' num2str(OptiumBeTa)]);

figure(6);% 在截层图像中显示迭代结果
Dx_init = Dx_MLE;
for t = 1:IL
    tic;
    disp(['BeTa = ' num2str(OptiumBeTa) ';' 'ICM iteration ' num2str(t) ' times...']); 
    Dx = ICM_estimation(VBN,pxl_k,inPLX,Object,Dx_init,OptiumBeTa,NB);
    subplot(1,IL,t);imshow(Dx(:,:,fix(c/2)),[]); pause(0.2);
    title(['MRF-ICM迭代' num2str(t) '次' ]); 
    Dx_init = Dx;
    ti = toc;disp(['Iteration runtime = ' num2str(ti)]);
end
vessel=OveralShows(SelecNum,Dx,Dx_MLE,flagIMG,Object);
disp('----------- All FINISHED -------------------------');


%**************** SubFunction ML_estimation **********************************
function [Dout,sort_D] = ML_estimation(A,VBN,Object,flagIMG)
% 函数意义：并行计算最大似然估计
% A,噪声条件下血管和背景混合数据；
% VBN3：各类均值和方差；
% W：类别比例系数
% sort_D:似然概率和头部索引IndexA的合成矩阵，有length(IndexA)行，Object+1列
tic;
disp('ML_estimation ...');
Dout = zeros(size(A));
IndexA = find(flagIMG~=0);%取被flagIMG标记为1的像素索引
% IndexA = Index_head;
N = numel(IndexA);
L = size(VBN,1);
A = repmat(A(IndexA)',L,1);                                                % A阵变为L行1列的A(:)'矩阵
mu = repmat(VBN(:,1),1,N);                                                 %产生L行Ni列的mu矩阵
sigma = repmat(VBN(:,2),1,N);                                              %产生L行Ni列的sigma矩阵
W = repmat(VBN(:,3),1,N);                                                  %产生L行Ni列的W矩阵
Li = find(1:L~=Object)';                                                   % 产生1至K-1行
pxl_k = [W(1,:).*((A(1,:)./mu(1,:))).*exp(-(A(1,:).^2./(2*mu(1,:)))); ...
         W(2:4,:).*(1./sqrt(2*pi*sigma(2:4,:).^2)).*exp(-(A(2:4,:)-mu(2:4,:)).^2./(2*sigma(2:4,:).^2))];
% DMPL_vector = Object*((pxl_k(Object,:)./W(Object,:))>(sum(pxl_k(Li,:),1)./sum(W(Li,:),1)));%第一种：权平均约束目标和背景类
DMPL_vector = Object*(pxl_k(Object,:)>max(pxl_k(Li,:),[],1));%第二种：血管类大于背景类的最大值
% DMPL_vector = Object*(pxl_k(Object,:)>sum(pxl_k(Li,:),1)/(L-1));%第三种：血管类大于背景类的均值,和上述第一种算法效果类似
Dout(IndexA)= DMPL_vector';%列矢量向输出矩阵赋值
index_PLX_object = [pxl_k' IndexA];% 针对flagIMG==1的空间体素，构造两列数组，第一列是pxl_k(Object)，第二列是体素的编号
sort_D = sortrows(index_PLX_object,-4);%在去脑壳后的空间中，按照目标类（pxl_k（:,4））由高至低的顺序输出矩阵
t = toc;
disp(['finished, time consumption of this step is ' num2str(t) ' s']);

%**************** SubFunction ICM_estimation **********************************
function [Dnew] = ICM_estimation(VBN,pxl_k,Ni,Object,D,beta,NB)
% A,噪声条件下血管和背景混合数据；
% flagIMG：腐蚀分区图D1_EM的空间（包含心脏）矩阵，排除了均值最低的两个类别，并用1代表对象空间，0代表背景空间
% D,参考标准数据，血管和背景分别为（4）和（3,2,1）；
% beta邻域作用系数；
W = VBN(:,3);
Li = 1:Object-1;%非目标类的序号
sizeD = size(D);
Dnew = zeros(sizeD);
[a,b,c] = ind2sub(sizeD,Ni);
s = [a b c];
FN = (NB == 0)*3 + (NB == 6)*1 +(NB == 26)*2;%自定义函数序号
f={@clique_6 @clique_26 @clique_MPN};%自定义势团函数
for n = 1:length(Ni)
    [pB,pV] = f{FN}(Object,s(n,:),D,beta);
    post_V = pV*pxl_k(n,4)/W (Object);
    post_B = pB*sum(pxl_k(n,Li'))/sum(W(Li'));
    Dnew(Ni(n)) = Object * (post_V >post_B);
end

%****************** SubFunction BeTa_estimation *************************
function [OptiumBeTa] = BeTa_estimation(IMG,VBN,flagIMG,Object,Dx_init,NB,MPL_alfa,Betaini)
% 利用降采样后的原始数据估计最优参数；
% 此时VBN中的比例参数变化较大，应对降采样后的数据重新估计参数
% 原始混合数据IMG、最大似然空间J、初始标记场Dx_init的尺寸都为：I_max*J_max*K_max
% BeTa，beta参数数组；
% M:奇偶交错为1的三维阵列;
% OptiumBeta,估计得到的最优Beta值
% MPL_alfa较大时，参数更新步长大，beta易在较大位置收敛（如：MPL_alfa=0.5时，收敛至0.8，MPL_alfa=0.1时，收敛至0.3）
% Betaini一般为零，但易产生局部极值，令其为0.3时可越过极值
[I_max,J_max,K_max] = size(IMG);
img = IMG(1:2:I_max,1:2:J_max,1:2:K_max);% 降采样IMG
[i_max,j_max,k_max] = size(img);% 降采样后的img尺寸
flagIMG = flagIMG(1:2:I_max,1:2:J_max,1:2:K_max);% 降采样J
Dx_init = Dx_init(1:2:I_max,1:2:J_max,1:2:K_max);% 降采样Dx_init
M = Neighbor_0_1([i_max,j_max,k_max]);% 产生坐标之和i+j+k奇偶交错为1的阵列
L = size(VBN,1);
mu = VBN(:,1);
sigma = VBN(:,2);
W = VBN(:,3);
Li = find(1:L~=Object);
N = cell(2,1);
N{1} = find(flagIMG==1 & M==0);%MRA数据空间奇点的索引
N{2} = find(flagIMG==1 & M==1);%MRA数据空间偶点的索引
P = cell(2,1);
[x,y,z] = ind2sub([i_max,j_max,k_max],N{1});%产生MRA数据空间奇点的坐标[x,y,z]
P{1} = [x,y,z];
[x,y,z] = ind2sub([i_max,j_max,k_max],N{2});%产生MRA数据空间奇点的坐标[x,y,z]
P{2} = [x,y,z];

FN = (3)*(NB==0)+1*(NB==6)+2*(NB==26);
f={@clique_6 @clique_26 @clique_MPN};

upNum = 6;
LnPL = zeros(upNum,1);LnPLk = zeros(2,1);
dMPL = zeros(upNum,1);dMPLk = zeros(2,1);
Beta = zeros(upNum,1);Beta(1) = Betaini;% 赋初值Betaini 
disp(['Betaini= ' num2str(Betaini)]);
for i = 2:upNum % 以下参数估计采用迭代计算公式，与MRA中的参数估计和run_ICM_phantom_1中的都不同
    for k = 1:2 
       Lc = length(N{k});
       sumVB_delta=eps;nV=eps;nB=eps;Eg=eps;
       for t = 1:Lc
           n = N{k}(t);            
           pxl_k = [(img(n)/mu(1))*exp(-img(n)^2/(2*mu(1)));...
                   (1./sqrt(2*pi*sigma(2:4).^2)).*exp(-(img(n)-mu(2:4)).^2./(2*sigma(2:4).^2))]; 
           s = P{k}(t,:);
           [pB,pV,NsigmaV,NsigmaB] = f{FN}(Object,s,Dx_init,Beta(i-1));
           fV = pxl_k(Object)+eps;
           fB = sum(W(Li).*pxl_k(Li))/sum(W(Li))+eps;
           post_V = fV * pV;
           post_B = fB * pB; 
           nV = nV + (post_V>post_B);
           nB = nB + (post_V <= post_B);
           BetaSigmaV = Beta(i-1)*NsigmaV;
           BetaSigmaB = Beta(i-1)*NsigmaB;
           expV = exp(-BetaSigmaV);
           expB = exp(-BetaSigmaB);
           %----------------计算伪似然能量函数1及其导数sumVB_delta     
           Eg = Eg + (post_V>post_B)*(- reallog(post_V));%伪似然能量函数1                              
           sumVB_delta = sumVB_delta + (post_V>post_B)*((NsigmaV*fV*expV + NsigmaB*fB*expB)/(fV*expV + fB*expB)-NsigmaV);%伪似然能量函数1的导数
       end
       LnPLk(k) = Eg/Lc;
       dMPLk(k) = sumVB_delta/nV;
    end
    LnPL(i) = mean(LnPLk);
    dMPL(i) = mean(dMPLk);
    dMPL(1) = 1.2*dMPL(2);%令dMPL(1)=1.2*dMPL(2);
    delta_dMPL = abs(dMPL(i))- abs(dMPL(i-1));
    alfa_i = (i==2)*MPL_alfa*abs(1/dMPL(2)) + (i>2)*MPL_alfa*abs((1/(abs(dMPL(2))-abs(dMPL(i))+eps)));
    Beta(i) = Beta(i-1) - (delta_dMPL<=0)*alfa_i*delta_dMPL;
    disp(['No.' num2str(i) '--- dMPL(' num2str(i) ')= ' num2str(dMPL(i)) '; alfa_i = ' num2str(alfa_i) '; Beta(' num2str(i) ')= ' num2str(Beta(i))]);
end
OptiumBeTa = Beta(i);
mindMPL = min(dMPL(2:upNum));maxdMPL = max(dMPL(2:upNum));deltaMPL = 0.5*abs(maxdMPL-mindMPL);
subplot(1,2,1);plot(1:upNum,Beta,'r');axis([1 upNum 0 1.2*max(Beta)]);xlabel('Iteration times');ylabel('beta');
subplot(1,2,2);plot(1:upNum,dMPL,'r');axis([2 upNum mindMPL-deltaMPL maxdMPL+deltaMPL]);xlabel('Iteration times');ylabel('Derivative of negative logarithm of PL');
% subplot(1,3,3);plot(1:upNum,LnPL,'k');axis([2 upNum 1.1*min(LnPL) 1.1*max(LnPL)]);xlabel('Iteration times');ylabel('Eg');
t3 = toc;
disp(['End of BeTa_estimation_1. The estimation time = ' num2str(t3) ' s']);

%****************** SubFunction cliqueMPN *************************
function [pB,pV,NsigmaV,NsigmaB] = clique_MPN(K,s,D,beta)
% 在均值水平mu(k)下，分别计算点s的属于血管目标的概率pV和属于背景的概率pB
% K为目标类的标记；A为待分割图像，D为初始标记场
% flag =0：代表s点及其邻域处于背景中；
% flag~=0：代表s点及其邻域处于目标中；
%% NsigmaV,目标点数；NsigmaB，背景点数
[i_max,j_max,n_max] = size(D);
i = s(1);j = s(2);n = s(3);
ip = (i+1<=i_max)*(i+1)+(i+1>i_max);im = (i-1>=1)*(i-1)+(i-1<1)*i_max;%----行加和减
jp = (j+1<=j_max)*(j+1)+(j+1>j_max);jm = (j-1>=1)*(j-1)+(j-1<1)*j_max;%----列加和减
np = (n+1<=n_max)*(n+1)+(n+1>n_max);nm = (n-1>=1)*(n-1)+(n-1<1)*n_max;%----层加和减
% 在既定的26邻域数组D_nb26中，定义空间邻域的矩阵结构
A = [5 11 13 15 17 23]; % D_nb26中的6邻域格点；
% 以(i,j,n)为中心，将26邻域的立方体由后向前、由左至右、由下至上分为8个立方体，每个立方体除(i,j,n)外有7个顶点,立方体的顶点标记值枚举如下：
C = [1 2 4 5 10 11 13;2 3 5 6 11 12 15;4 5 7 8 13 16 17;5 6 8 9 15 17 18;...
     10 11 13 19 20 22 23;11 12 15 20 21 23 24;13 16 17 22 23 25 26;15 17 18 23 24 26 27];
% 以(i,j,n)为中心，将26邻域的立方体由后向前、由左至右、由下至上分为6个六面体，每个六面体除(i,j,n)外有9个顶点,六面体的顶点标记值枚举如下：
F = [4 5 10 11 13 16 17 22 23;5 6 11 12 15 17 18 23 24;2 5 10 11 12 13 15 20 23; ...
     5 8 13 15 16 17 18 23 26;2 4 5 6 8 11 13 15 17;11 13 15 17 20 22 23 24 26]; 
 
D_nb26 = [D(im,jm,nm) D(i,jm,nm) D(ip,jm,nm) D(im,j,nm) D(i,j,nm) D(ip,j,nm) D(im,jp,nm) D(i,jp,nm) D(ip,jp,nm) ... %3x3x3立方体 -1层标记值
          D(im,jm,n)  D(i,jm,n)  D(ip,jm,n)  D(im,j,n)  D(i,j,n)  D(ip,j,n)  D(im,jp,n)  D(i,jp,n)  D(ip,jp,n) ...  %3x3x3立方体  0层标记值
          D(im,jm,np) D(i,jm,np) D(ip,jm,np) D(im,j,np) D(i,j,np) D(ip,j,np) D(im,jp,np) D(i,jp,np) D(ip,jp,np)];   %3x3x3立方体 +1层标记值          
% flag = sum(D_nb26(A)==K); % 统计中心点周围标记为K的点数(以空间6邻域为基准)
% 26邻域三维空间的N阶势团集合--------------------- 
NsigmaV_6 = sum(D_nb26(A)==K);NsigmaB_6 = sum(D_nb26(A)~=K);
NsigmaV_7 = max([sum(D_nb26(C(1,:))==K) sum(D_nb26(C(2,:))==K) sum(D_nb26(C(3,:))==K) sum(D_nb26(C(4,:))==K) ...
                     sum(D_nb26(C(5,:))==K) sum(D_nb26(C(6,:))==K) sum(D_nb26(C(7,:))==K) sum(D_nb26(C(8,:))==K)]);% C矩阵意义下的能量
NsigmaB_7 = max([sum(D_nb26(C(1,:))~=K) sum(D_nb26(C(2,:))~=K) sum(D_nb26(C(3,:))~=K) sum(D_nb26(C(4,:))~=K) ...
                     sum(D_nb26(C(5,:))~=K) sum(D_nb26(C(6,:))~=K) sum(D_nb26(C(7,:))~=K) sum(D_nb26(C(8,:))~=K)]);% C矩阵意义下的能量      
NsigmaV_9 = max([sum(D_nb26(F(1,:))==K) sum(D_nb26(F(2,:))==K) sum(D_nb26(F(3,:))==K) ...
                     sum(D_nb26(F(4,:))==K) sum(D_nb26(F(5,:))==K) sum(D_nb26(F(6,:))==K)]);   % F矩阵意义下的能量
NsigmaB_9 = max([sum(D_nb26(F(1,:))~=K) sum(D_nb26(F(2,:))~=K) sum(D_nb26(F(3,:))~=K) ...
                     sum(D_nb26(F(4,:))~=K) sum(D_nb26(F(5,:))~=K) sum(D_nb26(F(6,:))~=K)]);   % F矩阵意义下的能量       
% D_nb26中矩阵A表达的邻域的目标和背景能量
Uv6 = 6-NsigmaV_6; Ub6 = 6-NsigmaB_6;
% D_nb26中矩阵C表达的邻域的目标和背景能量
Uv_bound1 = 7 - NsigmaV_7; Ub_bound1 = 7 - NsigmaB_7;
Uv_bound2 = 9 - NsigmaV_9; Ub_bound2 = 9 - NsigmaB_9;
% 计算目标概率：无论D(i,j,n)是否为K，周围标记K较多时，能量最小，目标概率最大
Uvw = min([Uv6 Uv_bound1 Uv_bound2]);%Uvw = [Uv6 Uv_bound1 Uv_bound2];
Uv = beta*Uvw;
pV = exp(-Uv);
% 计算背景概率：当周围标记不为K的较多时，能量最小，背景概率最大
Ubw = min([Ub6 Ub_bound1 Ub_bound2]);%Ubw = [Ub6 Ub_bound1 Ub_bound2];
Ub = beta*Ubw;
pB = exp(-Ub);
% 计算目标点数NsigmaV和背景点数NsigmaB
NsigmaV = min([NsigmaV_6 NsigmaV_7 NsigmaV_9]);
NsigmaB = min([NsigmaB_6 NsigmaB_7 NsigmaB_9]);

%****************** SubFunction clique_26 *************************
function [pB,pV,NsigmaV,NsigmaB] = clique_26(K,s,D,beta)
% 在均值水平mu(k)下，分别计算点s的属于血管目标的概率pV和属于背景的概率pB
% K为目标类的标记；A为待分割图像，D为初始标记场
% flag =0：代表s点及其邻域处于背景中；
% flag~=0：代表s点及其邻域处于目标中；

% --------（1）----------计算邻域坐标，参考neighbouring2
[i_max,j_max,k_max] = size(D);
i = s(1);j = s(2);k = s(3);
%----行加1和减1
ip = (i<i_max)*(i+1)+(i==i_max);
im = (i>1)*(i-1)+(i==1)*i_max;
%----列加1和减1
jp = (j<j_max)*(j+1)+(j==j_max);
jm = (j>1)*(j-1)+(j==1)*j_max;
%----层加1和减1
kp = (k<k_max)*(k+1)+(k==k_max);
km = (k>1)*(k-1)+(k==1)*k_max;
% ---------End（1）------------
D_nb26 = [D(im,jm,km) D(i,jm,km) D(ip,jm,km) D(im,j,km) D(i,j,km) D(ip,j,km) D(im,jp,km) D(i,jp,km) D(ip,jp,km) ... %3x3x3立方体 -1层标记值
          D(im,jm,k)  D(i,jm,k)  D(ip,jm,k)  D(im,j,k)  D(i,j,k)  D(ip,j,k)  D(im,jp,k)  D(i,jp,k)  D(ip,jp,k) ...  %3x3x3立方体  0层标记值
          D(im,jm,kp) D(i,jm,kp) D(ip,jm,kp) D(im,j,kp) D(i,j,kp) D(ip,j,kp) D(im,jp,kp) D(i,jp,kp) D(ip,jp,kp)];   %3x3x3立方体 +1层标记值
% A = [5 11 13 15 17 23]; % D_nb26中的6邻域格点；
% flag = sum(D_nb26(A)==K); % 统计中心点周围标记为K的点数(以空间6邻域为基准)
NsigmaV = sum(D_nb26(1:27~=14)==K);
NsigmaB = sum(D_nb26(1:27~=14)~=K);
Uv26 = 26-NsigmaV;% 去除中心点14，D_nb26中矩阵A（=D_nb26）表达的26邻域的目标能量
Ub26 = 26-NsigmaB;% 去除中心点14，D_nb26中矩阵A（=D_nb26）表达的26邻域的背景能量
% 计算目标概率：无论D(i,j,k)是否为K，周围标记K较多时，能量最小，目标概率最大
Uv = beta*Uv26;
pV = exp(-Uv);
% 计算背景概率：当周围标记不为K的较多时，能量最小，背景概率最大
Ub = beta*Ub26;
pB = exp(-Ub);

%****************** SubFunction clique_6 *************************
function [pB,pV,NsigmaV,NsigmaB] = clique_6(K,s,D,beta)
% 在均值水平mu(k)下，分别计算点s的属于血管目标的概率pV和属于背景的概率pB
% K为目标类的标记；A为待分割图像，D为初始标记场
% fb记录点s与周围邻域标记的比较，不同则fb=1，相同则fb=0
% flag =0：代表s点及其邻域处于背景中；
% flag~=0：代表s点及其邻域处于目标中；
[i_max,j_max,n_max] = size(D);
i = s(1);j = s(2);n = s(3);
ip = (i+1<=i_max)*(i+1)+(i+1>i_max);im = (i-1>=1)*(i-1)+(i-1<1)*i_max;%----行加和减
jp = (j+1<=j_max)*(j+1)+(j+1>j_max);jm = (j-1>=1)*(j-1)+(j-1<1)*j_max;%----列加和减
np = (n+1<=n_max)*(n+1)+(n+1>n_max);nm = (n-1>=1)*(n-1)+(n-1<1)*n_max;%----层加和减
% 在既定的26邻域数组D_nb26中，定义空间邻域的矩阵结构
D_nb26 = [D(im,jm,nm) D(i,jm,nm) D(ip,jm,nm) D(im,j,nm) D(i,j,nm) D(ip,j,nm) D(im,jp,nm) D(i,jp,nm) D(ip,jp,nm) ... %3x3x3立方体 -1层标记值
          D(im,jm,n)  D(i,jm,n)  D(ip,jm,n)  D(im,j,n)  D(i,j,n)  D(ip,j,n)  D(im,jp,n)  D(i,jp,n)  D(ip,jp,n) ...  %3x3x3立方体  0层标记值
          D(im,jm,np) D(i,jm,np) D(ip,jm,np) D(im,j,np) D(i,j,np) D(ip,j,np) D(im,jp,np) D(i,jp,np) D(ip,jp,np)];   %3x3x3立方体 +1层标记值
A = [5 11 13 15 17 23]; % D_nb26中的6邻域格点； 
% flag = sum(D_nb26(A)==K); % 统计中心点周围标记为K的点数(以空间6邻域为基准)
NsigmaV = sum(D_nb26(A)==K);
NsigmaB = sum(D_nb26(A)~=K);
% 26邻域三维空间的N阶势团集合--------------------- 
Uv6 = 6-NsigmaV;% D_nb26中矩阵A表达的6邻域的目标能量
Ub6 = 6-NsigmaB;% D_nb26中矩阵A表达的6邻域的背景能量
% fb = (Uv6~=0);
% 计算目标概率：无论D(i,j,n)是否为K，周围标记K较多时，能量最小，目标概率最大
Uv = beta*Uv6;
pV = exp(-Uv);
% 计算背景概率：当周围标记不为K的较多时，能量最小，背景概率最大
Ub = beta*Ub6;
pB = exp(-Ub);

%****************** SubFunction Neighbor_0_1 *************************
function M = Neighbor_0_1(S)
M = zeros(S);
for i = 1:S(1)
    for j = 1:S(2)
        for n = 1:S(3)
            M(i,j,n)=(mod(i+j+n,2)==1);
        end
    end
end

%****************** SubFunction MRA_SegParameter1 ********************

function [VBN_EM,criValue,paraV] = SegParameter_MRA_pdf_curl(IMG,Iout,K,alfa,iterations)

% 函数意义：显示CTCA直方图、K均值初分类、计算各类百分比、在K均值类数前提下利用EM法精确估计参数
% 目标在于为MRF分割提供精确参数mu、sigma、w
% 显示截层图像和体数据住房图figure(1)、概率曲线拟合图figure(2)、参数迭代更新图figure(3)
% threthold = theta * criValue,criValue为临界阈值
close all
%%%%%%%%%%显示CTCA直方图
[a,b,c] = size(IMG);
img = IMG(1:2:a,1:2:b,1:2:c);% 矩阵降采样
LengthIMG = numel(img);
Max_img = max(img(:));
[~,~,c] = size(img);
figure(1);
subplot(1,2,1);imshow(imrotate(img(:,:,fix(c/4)),-90),[]);
[N,X] = hist(img(:),0:Max_img); 
%%%%%%%%%%%显示直方图上的极点
[Imax,Imin,N2] = peaks_Histogram(N);
hc = N2'/LengthIMG;
LN = length(hc);
subplot(1,2,2);
plot(1:LN,hc,'-b','LineWidth',2);hold on % 显示直方图曲线
plot(Imax,hc(Imax),'*r','MarkerSize',3);
plot(Imin,hc(Imin),'ob','MarkerSize',3);
axis([0 Max_img 0 max(hc)+0.1*max(hc)]);
grid on;axis square;hold off;
disp('[Imax(1) Imin Imax(2)] = ');
disp(num2str([Imax(1) Imin Imax(2)]));
%%%%%%%%%% K均值分类，并计算各类均值 K_mu、均方差K_var百分比K_percent
tic;
disp('kmeans...')
[idx,ctrs] = kmeans(img(:),K,'start',[Imax(1);Imin;Imax(2);350]);%4个初始分类点[Imax(1);Imin;Imax(2);350]
[Idx,Ctrs] = Kmean_reorder(idx,ctrs);% 按照灰度类中心由低至高的顺序重新输出idx和ctrs
K_mu = Ctrs;
Beta = Imax(1);%瑞丽表达混合成分的参数
K_var = zeros(K,1);% 方差
K_sigma = zeros(K,1);%标准差
Omega = zeros(K,1);%为混合成分占比
RG_K_curl = zeros(K,LN);
% figure(2);
% subplot(1,2,1);
figure;
plot(1:LN,hc,'-k','LineWidth',1.5);% 显示直方图曲线
axis([0 400 0 max(hc)+0.1*max(hc)]);grid on;hold on;
flag = {'-.b';'-.c';'-.m';'-g';'-r';'-k';'-.k';'-.y';'--k';':k';':g'};
for i = 1:K % 计算各类均方差K_var、百分比K_percent并绘高斯曲线图
    Omega(i) = length(find(Idx==i))/LengthIMG;% 各子类分布曲线的最大值
    K_var(i) = var(img(Idx==i));
    K_sigma(i) = sqrt(K_var(i));
    RG_K_curl(i,:) = (i==1)*Omega(i)*(X./Beta^2).*exp(-(X.^2./(2*Beta^2)))+...
                   (i~=1)*Omega(i)*(1/sqrt(2*pi)/K_sigma(i)).*exp(-(X-K_mu(i)).^2/(2*K_var(i)));%构建RGMM瑞丽高斯混合模型
    plot(1:LN,RG_K_curl(i,:),char(flag(i)),'LineWidth',1);%绘制各子类分布曲线
end
t = toc; disp(['using ' num2str(t) '秒']);
legend_char = cell(K+2,1);
legend_char{1} = char('Original histogram');
for i = 1:K % 编辑legend
    if i==1
        legend_char{1+i} = char(['Rayleigh curl-line' num2str(i) ': beta=' num2str(uint16(Beta))...
          ' w=' num2str(Omega(i))]);
    else
        legend_char{1+i} = char(['Gaussian curl-line' num2str(i) ': mu=' num2str(uint16(K_mu(i)))...
          ' sigma=' num2str(uint16(K_sigma(i))) ' w=' num2str(Omega(i))]);
    end
end
plot(1:LN,sum(RG_K_curl,1),'--r','LineWidth',1);% 显示拟合后的曲线
legend_char{K+2} = char('Init-fitting histogram');
legend(legend_char{1:K+2});
xlabel('Intensity');
ylabel('Frequency');
hold off

VBN_Init = [Beta^2     0         Omega(1);
            K_mu(2)  K_sigma(2)  Omega(2);
            K_mu(3)  K_sigma(3)  Omega(3);
            K_mu(4)  K_sigma(4)  Omega(4)];

% VBN_Init(4,1) = alfa*VBN_Init(4,1);%经验值 1.5*VBN_Init(4,1);
[criValue,paraV] = Iout_vessel_perscent(IMG,Iout,Omega(4),alfa); % threthold = theta * criValue,criValue为临界阈值

VBN_Rect = [Beta^2     0         Omega(1);
            K_mu(2)  K_sigma(2)  Omega(2);
            K_mu(3)  K_sigma(3)  Omega(3);
            paraV(1) paraV(2)    Omega(4)];

%%%%%%%%%%%%%%利用最大期望法精确估计以上各参数K_mean、K_sigma、K_percent
disp('RGMM_EM...');tic;
[VBN_EM, SumError] = RGMM_EM(IMG,VBN_Rect,iterations,0);%改用原始大尺寸数据IMG计算精确参数 %RGMM_EM(IMG,VBN_Init,1000,1);RGMM_EM(IMG,VBN_Rect,500,0)
% disp(['Finished, the curl_lines fitting error after EM step is: ' num2str(minError)]);
EM_mu = zeros(K,1);
EM_var = zeros(K,1);
EM_sigma = zeros(K,1);
Omega = zeros(K,1);
RG_EM_curl = zeros(K,LN);
% figure(3);
% subplot(1,2,2);
figure;
plot(1:LN,hc,'-k','LineWidth',1.5);% 显示直方图曲线
axis([0 400 0 max(hc)+0.1*max(hc)]);grid on;hold on;
for i = 1:K % 计算各类均值K_mean、均方差K_sigma百分比K_percent
    EM_mu(i) = VBN_EM(i,1);
    Omega(i) = VBN_EM(i,3);% 各子类分布曲线的最大值
    EM_var(i) =  VBN_EM(i,2)^2+eps(1);
    EM_sigma(i) = VBN_EM(i,2)+eps(1);
    RG_EM_curl(i,:) = (i==1)*Omega(i)*(X./EM_mu(i,1)).*exp(-(X.^2./(2*EM_mu(i,1))))+...
                   (i~=1)*Omega(i)*(1/sqrt(2*pi)/EM_sigma(i)).*exp(-(X-EM_mu(i)).^2/(2*EM_var(i))); 
    plot(1:LN,RG_EM_curl(i,:),char(flag(i)),'LineWidth',1);
end
t = toc; disp(['using ' num2str(t) '秒']);
legend_char = cell(K+2,1);
legend_char{1} = char('Original histogram');
for i = 1:K % % 编辑legend
    if i==1
       legend_char{1+i} = char(['EM Rayleigh curl-line ' num2str(i) ': beta=' num2str(uint16(sqrt(EM_mu(i))))...
           ' w=' num2str(Omega(i))]);
    else
       legend_char{1+i} = char(['EM Gaussian curl-line ' num2str(i) ': mu=' num2str(uint16(EM_mu(i)))...
           ' sigma=' num2str(uint16(EM_sigma(i))) ' w=' num2str(Omega(i))]);
    end
end
plot(1:LN,sum(RG_EM_curl,1),'--r','LineWidth',1);% 显示拟合后的曲线
legend_char{K+2} = char('EM fitting histogram');
legend(legend_char{1:K+2});
xlabel('Intensity');
ylabel('Frequency');
hold off

%----输出Kmeans的参数估计结果
VBN_Initshow = VBN_Init;
VBN_Initshow(1,1) = Beta;
disp('VBN_Init =');disp(num2str(VBN_Initshow));
disp(['VBN_Init by KmeansSize: [' num2str(size(img)) ']']);
%----输出修正后的参数估计结果
VBN_Rectshow = VBN_Rect;
VBN_Rectshow(1,1) = Beta;
disp('VBN_Rect =');disp(num2str(VBN_Rectshow));
disp(['VBN_Init by RectSize: [' num2str(size(IMG)) ']']);
%----输出最大期望参数估计结果
VBN_EM_show = VBN_EM; 
VBN_EM_show(1,1) = sqrt(VBN_EM(1,1));
disp('VBN_EM =');disp(num2str(VBN_EM_show));
disp(['VBN_EM by EMSize: [' num2str(size(IMG)) ']']);
%---------------
disp(['All over. The EM Fitting error is ' num2str(SumError)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% 子函数1%%%%%%%%%%%%%%%%%%
function [Imax,Imin,N2] = peaks_Histogram(N)%======可以改进一下。
% 查找3D-CTCA直方图曲线的峰值点和谷值点Imax,Imin
N2 = smooth(N,32,'loess'); %21
LN = length(N2);
DN2 = diff([N2;N2(LN)]);
n1 = 0;
n2 = 0;
I_max=[];
I_min=[];
I_MIN=[];
for i = 5:200 % 从第10个点开始 200=>length(N2)-2
    if ((DN2(i-1)>0) && (DN2(i+1)<0)) && N2(i)>max(N2(i-1),N2(i+1)) && N2(i)>10
       n1 = n1+1;
       I_max(n1) = i;
    end
    if ((DN2(i-1)<0) && (DN2(i+1)>0)) && N2(i)<min(N2(i-1),N2(i+1))
       n2 = n2+1;
       I_min(n2) = i;
    end 
end
% 在各个峰值点I_max之间找出最优的谷点
n_max = length(I_max); % 峰值点数
for k = 1:n_max
    if k ~= n_max
       nums = find(I_min>I_max(k) & I_min<I_max(k+1));
       [~,m] = min(N2(I_min(nums)));
       I_MIN(k) = I_min(nums(m));
    end
    if k == n_max && ~isempty(find(I_min>I_max(k)))
       nums = find(I_min>I_max(k));
       [~,m] = min(N2(I_min(nums)));
       I_MIN(k) = I_min(nums(m));
    end
end
Imax = I_max;
Imin = I_MIN;

%**************** SubFunction Iout_vessel_perscent ***************************
function [criValue,paraV] = Iout_vessel_perscent(IMG,Iout,percentEM,alfa)
disp('computing the optiumal threthold,mu4,sigma4 ...');
[y,x,z] = size(Iout);
b = numel(Iout);
N = 100;
MultiplyNum = 10;% 人为乘数(不影响分割结果)——以便subplot(1,2,1)中显示的曲线变为血管与头颅容积比例，MultiplyNum=1为客观真实值
Iout =Iout/max(Iout(:));%归一化为[0,1]
thre_value = linspace(0.001,0.08,N);
ratio = zeros(N,1);
for i = 1:N
    Gi = GetRegion(Iout,thre_value(i)); % 输出阈值thre_value(i)下的最大血管分支块
    ratio(i) = MultiplyNum*length(find(Gi>thre_value(i)))/b;
end
[~,numr] = min(abs(ratio-MultiplyNum*percentEM));
figure(3);
subplot(1,2,1);plot(thre_value,ratio,'-b',thre_value,ratio(numr)*ones(1,N),'-.r',thre_value(numr),ratio(numr),'*r');
xlabel('threshold values');ylabel('ratios')
legend('ratio curve of cerebral vessel to head volume','ratio given by prior knowledge');
axis([min(thre_value) max(thre_value) 0 MultiplyNum*0.005])

criValue = thre_value(numr);                 % criValue作为临界点，可用来计算血管的初始参数,后续利用gama*criValue产生血管试探空间
Stemp = GetRegion(Iout,criValue)>criValue;   % 血管初始空间
IMG_mu4 = mean(IMG(Stemp));                  % IMG中的血管均值
IMG_sigma4 = std(IMG(Stemp));                % IMG中的血管标准差
paraV = [alfa*IMG_mu4,IMG_sigma4];

% 显示'w4'对应的最大血管分支块
Gout = GetRegion(Iout,thre_value(numr));% 输出'w4'对应的血管分支块
subplot(1,2,2);patch(isosurface(Gout,0.5),'FaceColor','r','EdgeColor','none');
axis([0 x 0 y 0 z]);view([270 270]); daspect([1,1,1]);
camlight; camlight(-80,-10); lighting phong; pause(1); 
title('thresholding the multi-scale filtering response at the appointed ratio');
% view([0 -90]);
% view([270 270]); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%子函数2%%%%%%%%%%%%%%%%%

function [VBN, SumError] = RGMM_EM(B,Init,upNum,Flag)
% Flag=1:显示参数迭代更新图figure(3)，否则不显示
% upNum:迭代上限数值
% Init：迭代之前VBN的初始参数集

K = size(Init,1);
N = length(B(:));
B_max = max(B(:));
[fc,xi] = hist(B(:),0:B_max);
L = length(0:B_max);

FC = M_Extand(fc,K,L);
XI = M_Extand(xi,K,L);

Mu = zeros(K,upNum+1);
Var = zeros(K,upNum+1);
W = zeros(K,upNum+1);
Error = zeros(1,upNum);
dL = length(Error);

Mu(:,1) = Init(:,1);
Var(:,1) = Init(:,2).^2;
W(:,1) = Init(:,3);

for i = 1:upNum
    [plx,pxl] = pLX(0:B_max,K,Mu(:,i),Var(:,i),W(:,i));
    W(:,i+1) = (1/N)*sum(FC.*plx,2);
    Mu(1,i+1) = sum(XI(1,:).^2.*FC(1,:).*plx(1,:),2)./(2*sum(FC(1,:).*plx(1,:),2));% (1,:)对应瑞利分布密度函数
%     Mu(2:K,i+1) = sum(XI(2:K,:).*FC(2:K,:).*plx(2:K,:),2)./sum(FC(2:K,:).*plx(2:K,:),2);% 区间(2:K,:)对应的高斯均值
    Mu(2:K-1,i+1) = sum(XI(2:K-1,:).*FC(2:K-1,:).*plx(2:K-1,:),2)./sum(FC(2:K-1,:).*plx(2:K-1,:),2);% 区间(2:K-1,:)对应的高斯均值
    Mu(K,i+1) = Mu(K,i);% 区间K对应的高斯均值
    MU = M_Extand(Mu(:,i+1),K,L);
    Var(2:K,i+1) = sum((XI(2:K,:)-MU(2:K,:)).^2.*FC(2:K,:).*plx(2:K,:),2)./sum(FC(2:K,:).*plx(2:K,:),2);
    Error(i) = sum(abs(sum(pxl,1)-fc/N));
end
VBN = [Mu(:,upNum+1) sqrt(Var(:,upNum+1)) W(:,upNum+1)];
% [minError,i_num] = min(Error);
% VBN_EM = [Mu(:,i_num+1) sqrt(Var(:,i_num+1)) W(:,i_num+1)];

if Flag==1
figure(3);
legend_char = cell(K+1,1);
subplot(1,3,1);plot(1:dL,Mu(:,1:dL));
for k = 1:K
    if k==1
       legend_char{k} = char(['beta updates from ' num2str(sqrt(Mu(k,1))) ' to ' num2str(sqrt(Mu(k,dL)))]);
    else
       legend_char{k} = char(['mu' num2str(k-1) ' updates from ' num2str(Mu(k,1)) ' to ' num2str(Mu(k,dL))]);
    end
end
legend(legend_char{1:K});
axis([0 dL+1 0 1.5*max(Mu(:))+20]);
xlabel('Times of EM iteration');
ylabel('Mean of each classification')

subplot(1,3,2);plot(1:dL,sqrt(Var(:,1:dL)));
for k = 2:K
    legend_char{k} = char(['sigma' num2str(k-1) ' updates from ' num2str(sqrt(Var(k,1))) ' to ' num2str(sqrt(Var(k,dL)))]);
end
legend(legend_char{2:K});
axis([0 dL+1 0 1.5*max(sqrt(Var(:)))+10]);
xlabel('Times of EM iteration');
ylabel('Sigma of each classification')

subplot(1,3,3);plot(1:dL,W(:,1:dL));
for k = 1:K
    legend_char{k} = char(['w' num2str(k) ' updates from ' num2str(fix(100*W(k,1))/100) ' to ' num2str(fix(100*W(k,dL))/100)]);
end
legend(legend_char{1:K});
axis([0 dL+1 0 1.2]);
xlabel('Times of EM iteration');
ylabel('Weight of each classification')

figure(4);
plot(1:dL,Error(1:dL));
axis([0 dL 0 max(Error)]);
xlabel('Times of EM iteration');
ylabel('MSE of the parameters between neiboring iteration')

end
SumError = Error(dL);

%***********************************************************************************
%************** 子函数：求后验概率,f(k|xi) = wk*f(xi|k)/∑j=1:K(wj*f(xi|j))*********
function [plx,pxl] = pLX(xi,K,Mu,Var,W)
% 计算后验概率矩阵plx
pxl = zeros(K,length(xi));% 初始化条件概率矩阵
plx = zeros(K,length(xi));% 初始化后验概率矩阵
Var = Var + eps(1);       % 使得第一行方差为不等于零的最小数，以免1/sqrt(2*pi*Var(k))为NaN
for k = 1:K               % 逐类别计算条件概率矩阵
    pxl(k,:) = (k==1)*W(k)*(xi./Mu(1)).*exp(-(xi.^2./(2*Mu(1)))) + ...
               (k~=1)*W(k)*(1/sqrt(2*pi*Var(k)))*exp(-(xi-Mu(k)).^2./(2*Var(k)));
end
Sum_pxl = sum(pxl,1)+eps(1);
for k = 1:K
    plx(k,:) = pxl(k,:)./Sum_pxl;
end

function [D] = M_Extand(Vector,K,L)
% size(Vector) = [K,1] or [1,L]
% 将Vector扩展为矩阵D，size(D)=[K,L]
D = zeros(K,L);
[a,b] = size(Vector);
if a>1 && b==1 %如果Vector是一列矢量K×1
   for j = 1:L
       D(:,j) = Vector;
   end    
end
if a==1 && b>1 %如果Vector是一行矢量1×L
    for i = 1:K
        D(i,:) = Vector;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%子函数4 %%%%%%%%%%%%%%%%%%

function [Idx,Ctrs] = Kmean_reorder(idx,ctrs)
% 函数意义：将idx和ctrs中的内容按照ctrs由小到大的顺序重新排序
K = length(ctrs);
Ctrs_index = [ctrs,(1:length(ctrs))'];
sort_Ctrs = sortrows(Ctrs_index,1);% sort_Ctrs(:,1)由低至高存放聚类中心；sort_Ctrs(:,2)存放对应的原始类别号k
K_index = cell(1,K);
for k = 1:K % 将idx中原始分类号k分别存储在K_index结构中
    K_index{k} = find(idx==k);
end
Idx = zeros(size(idx));
Ctrs = zeros(size(ctrs));
for i = 1:K
    Ctrs(i) = sort_Ctrs(i,1);
    Idx(K_index{sort_Ctrs(i,2)}) = i;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% imshow_flagIMG %%%%%%%%%%%%%%%
function imshow3D_patch(D,D_original,colormode)
% D与D_original具有相同尺寸；且[a,b,c] = size(D)
% 获取3D二值数据空间D_original（0/1数值）尺寸，并在此空间上显示D中目标；
index = find(D_original==1);
[a,b,c] = ind2sub(size(D_original),index);
mina = min(a);maxa = max(a);
minb = min(b);maxb = max(b);
minc = min(c);maxc = max(c);
F = zeros((maxa-mina+1)+9,(maxb-minb+1)+9,(maxc-minc+1)+9);
F(4:4+(maxa-mina),4:4+(maxb-minb),4:4+(maxc-minc)) = D(mina:maxa,minb:maxb,minc:maxc);
[a,b,c] = size(F);
patch(isosurface(F,0.5),'FaceColor',colormode,'EdgeColor','none');
axis([0 b 0 a 0 c]);view([270 270]);daspect([1,1,1]);
camlight; camlight(-80,-10); lighting phong; pause(1); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% GetRegion %%%%%%%%%%%%%%%
function [Gout,MaxLength] = GetRegion(Iin,threthold,SelecNum)
% 将Iin二值化后，提取所有目标块，按照索引找出最大块序号Max_num，输出最大块区域Iout（二值化）
% T提取最大块的次数，第k次提取k-1次选剩的结果
% Tt为[1 2 ... T]的成员数组，代表想提取目标块的序号
J = Iin>threthold;
sIdx  = regionprops(J,'PixelIdxList');      % 提取J中所有目标块区域的像素索引；
num_sIdx = zeros(length(sIdx),1);           % 存放sIdx长度的矩阵num_sIdx
Gout = zeros(size(J));                      % 新建立相同尺寸的标记矩阵Iout
for i = 1:length(sIdx)
    num_sIdx(i) = length(sIdx(i).PixelIdxList);
end
if nargin ==2                               % 当输入前两个参数时
   [~,Max_num] = max(num_sIdx);             % 输出矩阵num_sIdx中的最大长度序号
   MaxPatch = sIdx(Max_num).PixelIdxList;
   Gout(MaxPatch) = 1;                      % 输出最大块
   MaxLength = length(MaxPatch);            % 输出最大块的长度
   return;
end
MaxLength = [];
maxSN = max(SelecNum);                      % 选择的最大块的最大序号
for t = 1:maxSN                             % 寻找指定最大块
    [~,Max_num] = max(num_sIdx);            % 最大块的长度    
    num_sIdx(Max_num)=0;                    % 将num_sIdx中找到的最大块的长度置0，以便后续寻找次大值            
   if ismember(t,SelecNum)                  % 如果为指定的最大块，则输出之
      MaxPatch = sIdx(Max_num).PixelIdxList;
      Gout(MaxPatch) = 1;    % 形成最大似然空间（像素1代表）
   end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% imshow_Process %%%%%%%%%%%%%%
function imshow_Process(IMG,Iout,Dx_init)
c = size(IMG,3);
subplot(1,4,1);imshow(imrotate(IMG(:,:,fix(c/2)),-90),[]);title('Original Image');
subplot(1,4,2);imshow(imrotate(squeeze(max(IMG,[],3)),-90),[]);title('MIP of Original Image');
subplot(1,4,3);imshow(imrotate(squeeze(max(Iout,[],3)),-90),[]);title('MIP after Vessel Enhance');
subplot(1,4,4);imshow(imrotate(Dx_init(:,:,fix(c/2)),-90));title('ML_estimation');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% OveralShows %%%%%%%%%%%%%%
function [vessel]=OveralShows(SelecNum,Dx,Dx_MLE,flagIMG,Object)
Dout = zeros([size(Dx) 2]);% Dout为0/4二值矩阵
[Dout(:,:,:,1)] = GetRegion(Dx,0,SelecNum);% 逐次输出第SelecNum(i)次最大的血管分支   
[Dout(:,:,:,2)] = GetRegion(Dx,0,[1 2 3]);% 最大的血管分支  
figure;
subplot(1,3,1);imshow3D_patch(Dx_MLE,flagIMG,[1 0 0]);title('ML估计');
subplot(1,3,2);imshow3D_patch(Object*Dout(:,:,:,1),flagIMG,[1 0 0]);title('Markov最大后验估计');
subplot(1,3,3);imshow3D_patch(Object*Dout(:,:,:,2),flagIMG,[1 0 0]);title('最大连通区域');

%vessel=zeros([size(Dx) 1]);
vessel=Dout(:,:,:,2);
disp(['Length of cerebral vessel is ' num2str(length(find(Dout(:,:,:,2)==1))) ' voxels']);


