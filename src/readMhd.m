function [Image, DimSiz,ElementSpace,info]=readMhd(mhdfile_path)
DimSiz=[];
ElementSpace=[];
[filepath,~,~]=fileparts(mhdfile_path);
mhd_fid = fopen(mhdfile_path);
if(mhd_fid<0)
    fprintf('could not open this file');
    return
end
while ~feof(mhd_fid)
     str=fgetl(mhd_fid);
    s=find(str=='=',1,'first');
    if(~isempty(s))
        type=str(1:s-1); 
        data=str(s+1:end);
        while(type(end)==' '); type=type(1:end-1); end
        while(data(1)==' '); data=data(2:end); end
    else
        type =''; data =str;
    end
    switch(lower(type))
        case 'ndims'
            info.NumberOfDimensions=sscanf(data, '%d')';
        case 'dimsize'
            info.Dimensions=sscanf(data, '%d')';
            DimSiz =  info.Dimensions;
        case 'elementspacing'
            info.PixelDimensions=sscanf(data, '%lf')';
            ElementSpace =  info.PixelDimensions;
        case 'elementsize'
            info.ElementSize=sscanf(data, '%lf')';
            if(~isfield(info,'PixelDimensions'))
                info.PixelDimensions=info.ElementSize;
            end
        case 'elementbyteordermsb'
            info.ByteOrder=lower(data);
        case 'anatomicalorientation'
            info.AnatomicalOrientation=data;
        case 'centerofrotation'
            info.CenterOfRotation=sscanf(data, '%lf')';
        case 'offset'
            info.Offset=sscanf(data, '%lf')';
        case 'binarydata'
            info.BinaryData=lower(data);
        case 'compresseddatasize'
            info.CompressedDataSize=sscanf(data, '%d')';
        case 'objecttype',
            info.ObjectType=lower(data);
        case 'transformmatrix'
            info.TransformMatrix=sscanf(data, '%lf')';
        case 'compresseddata';
            info.CompressedData=lower(data);
        case 'binarydatabyteordermsb'
            info.ByteOrder=lower(data);
        case 'elementdatafile'
            info.DataFile=data;
            readelementdatafile=true;
        case 'elementtype'
            info.DataType=lower(data(5:end));
        case 'headersize'
            val=sscanf(data, '%d')';
            if(val(1)>0), info.HeaderSize=val(1); 
            end
        otherwise
            info.(type)=data;
    end
end
if readelementdatafile
    image_file = fullfile(filepath,info.DataFile);
    fid = fopen(image_file,'r');
    clear img
    switch(info.DataType)
        case 'ushort'
            img = fread(fid,'uint16');
        case 'float'
            img = fread(fid,'float');
        otherwise
            img = fread(fid,'uint16');
    end
    fclose(fid);
    Siz = info.Dimensions;
    Image = double(reshape(img,Siz(1),Siz(2),Siz(3)));
end
