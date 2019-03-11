function [Trans_Image] = Hist_match3D(Target_Hist,Image)
%HIST_MATCH3D �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%           FOR TOF-MRA 
% Image: TOF-MRA without kull ���任��ͼ��
% Targe_Hist: Ŀ��ֱ��ͼ�ֲ�
% Trans_Image���任���ͼ��

%calculate the hist of the Image(���任��ͼ��)
img = Image((Image>=1));
LengthIMG = numel(img);
Max_intensity = max(img(:));
Test_Hist=zeros(floor(Max_intensity)+1,1);
Hist_Cell=cell(floor(Max_intensity)+1,1);
tic;
for i=1:length(Test_Hist)
    Hist_Cell{i}=img(img>=(i-1) & img<i);
    Test_Hist(i)=numel(Hist_Cell{i});
end
Test_Hist = Test_Hist/ LengthIMG;
t=toc;
disp(['calculate the hist of the Image---runtime = ' num2str(t)]);pause(0.1);
figure
subplot(2,2,1)
X=0:1:length(Target_Hist)-1;
plot(X,Target_Hist,'-b','LineWidth',2);
subplot(2,2,2)
X=0:1:length(Test_Hist)-1;
plot(X,Test_Hist,'-r','LineWidth',2);


% calculate  the cumulative histogram

Target_lens=length(Target_Hist);
C_target_hist=[];               %Ŀ���ۻ��ֲ�
for i=1:Target_lens
   C_target_hist=[C_target_hist sum(Target_Hist(1:i))]; 
end
subplot(2,2,3)
plot(0:Target_lens-1,C_target_hist,'-b','LineWidth',2);

Test_lens=length(Test_Hist);
C_test_hist=[];               %���任���ݵ��ۻ��ֲ�
for i=1:Test_lens
   C_test_hist=[C_test_hist sum(Test_Hist(1:i))]; 
end
subplot(2,2,4)
plot(0:Test_lens-1,C_test_hist,'-r','LineWidth',2);
suptitle('Left: Target-hist,Right:Pre trans-hist');pause(0.1);
% transfer the hist of the image to specific hist
tic;
Map_table=zeros([Target_lens,3]);
tick=0;
for i = 1:Target_lens%
    Map_table(i,1)=tick;% �任��Ӧ�����޻Ҷ�ֵ
    temp=C_test_hist-C_target_hist(i);
    [~,index1]=max(temp(find(temp<=0))); % ��С����ӽ���
    if length(index1)>1
        index1=index1(end);
    end
    index2=min(index1+1,Test_lens);% find �ϴ����ӽ��ķ����
    new_V=C_test_hist(index1);
    while(Test_Hist(index2)==0)
        index2=index2+1;
        if index2==Test_lens
            break;
        end
    end
    A=Test_Hist(index2);% �ϴ����ӽ��ķ���������еĻҶ�ֵ
    B=C_target_hist(i)-C_test_hist(index1);
    Ratio=B/A*100;
    if Ratio==0
        tick=index2-1-1e-10;
    else
        tick = prctile(Hist_Cell{index2},B/A*100);
        new_V=new_V+numel(find(Hist_Cell{index2}>=tick))/LengthIMG;
    end
    Map_table(i,2)=tick;% �任��Ӧ�����޻Ҷ�ֵ
    Map_table(i,3)=new_V; % �任���Ӧ���ۼƷֲ�����
end
t=toc;
disp(['transfer the hist of the image to specific hist---runtime = ' num2str(t)]);pause(0.1);
% end ��ɶ�Ӧ�任��

% Image transformation based on transformation table
tic;
Trans_Image=zeros(size(Image));
for i=1:Target_lens
    inf_V=Map_table(i,1);
    sup_V=Map_table(i,2);
    Trans_Image(Image>inf_V & Image<=sup_V)=i-1;
end
t=toc;
disp(['Image transformation---runtime = ' num2str(t)]);pause(0.1);
img = Trans_Image((Trans_Image>0));
LengthIMG = numel(img);
Max_img = max(img(:));
[~,~,c] = size(Trans_Image);
[N,~] = hist(img(:),0:Max_img); 
Trans_Hist=N'/LengthIMG;

figure;
subplot(2,2,1);imshow(imrotate(Image(:,:,fix(c/2)),-90),[]);
subplot(2,2,2);plot(0:length(Test_Hist)-1,Test_Hist,'-b','LineWidth',2);
subplot(2,2,3);imshow(imrotate(Trans_Image(:,:,fix(c/2)),-90),[0,400]);
subplot(2,2,4);plot(0:length(Trans_Hist)-1,Trans_Hist,'-b','LineWidth',2);
suptitle('Up: Original Image and hist| Down: Tranfered Image and hist');pause(0.1)

end
