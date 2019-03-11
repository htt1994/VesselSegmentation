clear;
clc;
close all;
numid =1;

%https://itk.org/Wiki/Proposals:Orientation#Some_notes_on_the_DICOM_convention_and_current_ITK_usage
% LOAD TOF-MRA DATA
MetaDataFilepath =strcat( '..\data\Tof_',num2str(numid,'%03d'),'_corr.mhd');

% Read ImageData and dataInfo
[Image, DimSiz,ElementSpace,info] = readMhd(MetaDataFilepath);

Offset = info.Offset;
TransformMatrix =  reshape(info.TransformMatrix,[3,3])';

% Multi-scalar vessel enhancement
I = Image - min(Image(:));
I = I / prctile(I(I(:) > 0.5 * max(I(:))),90);
I(I>1) = 1;
Iout = vesselness3D(I, 1:2,[1;1;1], 0.7, true);
Iout =Iout - min(Iout(:));
Iout = Iout /max(Iout(:));

% Get some high-confidence vessel points
hc_value = prctile(Iout(Iout>0),99);
Initial_Vp=(Iout >= hc_value);
% show these initial vessel points
figure;
patch(isosurface(Initial_Vp,0.5),'FaceColor',[1,0,0],'EdgeColor','none')
[a,b,c] = size(Initial_Vp);axis([0 b 0 a 0 c]);view([270,270]);
daspect([1,1,1]);
title('Initial Vessel Points');camlight; camlight(-80,-10); lighting phong;

% fuse Iout and Image 
FImage =Iout.*prctile(Image(Image>0),95)*0.5+(Iout.*(Iout<=hc_value)./hc_value+double(Iout>hc_value)).*Image*0.5;
%FImage =Iout.*prctile(Image(Image>0),95);
%show the Histogram of FImage,(without zero-value considering)
figure;
img = FImage((FImage>0));
% img = FImage;
[N_img,X_img] = hist(img(:),0:max(img(:))); 
hc_img = N_img'/numel(img);
LN_img = length(hc_img);
plot(0:max(img(:)),hc_img,'-r');

% begin the process of histogram matching 
load('Target_Hist.mat')
[Trans_Image] = Hist_match3D(Target_Hist,FImage);

VesselPs=find(Initial_Vp>0);
rand_index = randperm(length(VesselPs));
draw_rand_index = rand_index(1:round(length(VesselPs)));
VesselPoint_Matlab=[];
[VesselPoint_Matlab(:,1),VesselPoint_Matlab(:,2),VesselPoint_Matlab(:,3)] = ind2sub(size(Initial_Vp),VesselPs(draw_rand_index));

[Dx,vessel,vessel_ratio ]=GMM_MRF_SEMI(Trans_Image,Iout,3,3,[1:20],6,0.5,3,500,VesselPoint_Matlab);

figure;
set(gca,'color','black'); 
[a,b,c] = size(vessel);
[Vessel_maxConnect, ~,~] = Connection_Judge_3D(vessel,3,[1 2 4 5 6 9],10,2);
patch(isosurface(Vessel_maxConnect,0.5),'FaceColor',[1,0,0],'EdgeColor','none');
axis([0 b 0 a 0 c]);view([270,270]);
daspect([1 1 1]);
title(['SMPM提取结果Normale',num2str(numid,'%03d')]);camlight; camlight(-80,-10); lighting phong; 

%================分界线==============================
% A=FImage.*Vessel_maxConnect;

% VesselPs=find(Initial_Vp>0);
% rand_index = randperm(length(VesselPs));
% draw_rand_index = rand_index(1:round(length(VesselPs)));
% VesselPoint_Matlab=[];
% 
% [VesselPoint_Matlab(:,1),VesselPoint_Matlab(:,2),VesselPoint_Matlab(:,3)] = ind2sub(size(Initial_Vp),VesselPs(draw_rand_index));

% [Dx,vessel,vessel_ratio ]=GMM_MRF_SEMI(Trans_Image,Iout,3,3,[1:10],6,0.5,3,500,VesselPoint_Matlab);

figure;
set(gca,'color','black'); 
[a,b,c] = size(vessel);
[Vessel_maxConnect, C_number,~] = Connection_Judge_3D(vessel,2,[1],10,1);
patch(isosurface(Vessel_maxConnect,0.5),'FaceColor',[1,0,0],'EdgeColor','none');
axis([0 b 0 a 0 c]);view([270,270]);
daspect([1 1 1]);
title(['SMPM提取结果Normale',num2str(numid,'%03d')]);camlight; camlight(-80,-10); lighting phong; 

% % Improve the Continuity of Intensity 
% g_keranl = fspecial3('gaussian',3,0.4);
% FImage=imfilter(Image,g_keranl);
% [a,b,c]=size(Image);
% Image = imresize3(Image,[2*a,1.5*b,4*c]);

% [a,b,c]=size(Image);
% img = Image((Image>0));
% 
% figure;
% subplot(1,2,1);
% imshow(imrotate(Image(:,:,fix(c/2)),-90),[]);
% [N_img,X_img] = hist(img(:),0:max(img(:))); 
% hc_img = N_img'/numel(img);
% LN_img = length(hc_img);
% subplot(1,2,2);
% plot(1:LN_img,hc_img,'-r');hold on % 显示直方图曲线
% axis([0 450 0 max(hc_img)]);grid on;axis square;hold off;

% begin the process of histogram matching 
% load('Target_Hist.mat')
% [Trans_Image] = Hist_match3D(Target_Hist,FImage);