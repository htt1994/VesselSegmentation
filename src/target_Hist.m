clc;clear;
load('Normal002.mat', 'Image');
img = Image((Image~=0));
LengthIMG = numel(img);
Max_img = max(img(:));
[~,~,c] = size(Image);
figure(1);
subplot(1,2,1);imshow(imrotate(Image(:,:,fix(c/2)),-90),[]);
[N,X] = hist(img(:),0:Max_img); 
hc=N'/LengthIMG;
LN = length(hc);
subplot(1,2,2);
plot(1:LN,hc,'-b','LineWidth',2);hold on % 显示直方图曲线
axis([0 400 0 max(hc)+0.1*max(hc)]);
grid on;axis square;hold off;
Target_Hist = hc;
save('Target_Hist.mat','Target_Hist');