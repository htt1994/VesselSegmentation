function [Imax,Imin,N2] = peaks_Histogram(N)%======可以改进一下。
% 查找3D-CTCA直方图曲线的峰值点和谷值点Imax,Imin
N2 = smooth(N,10,'loess'); %21
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
end