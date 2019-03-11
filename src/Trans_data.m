% numid=1;
% DataFilepath =strcat( 'C:\Users\17488\Desktop\tof_space\TOF-1\tof',num2str(numid,'%03d'),'.nii');
% Data=load_untouch_nii(DataFilepath);
% Image=Data.img;
% nii_Image=make_nii(Image);
% savepath =strcat( 'C:\Users\17488\Desktop\tof_space\TOF-1\Re_tof',num2str(numid,'%03d'),'.nii');
% save_nii(nii_Image,savepath);


DataFilepath =strcat( 'C:\Users\17488\Desktop\tof_space\TOF-1\New_tof001.nii');
Data=load_untouch_nii(DataFilepath);
Image=Data.img;
Image=int16(Image);
nii_Image=make_nii(Image);
savepath =strcat( 'C:\Users\17488\Desktop\tof_space\TOF-1\New_tof2001.nii');
save_nii(nii_Image,savepath);