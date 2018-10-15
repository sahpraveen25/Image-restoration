A=imread('kernel.png');
img=imread('Blurry1_1.jpg');
originalimg=double(imread('Groundtruth1_1_1.jpg'));
red = img(:,:,1); % Red channel
green = img(:,:,2); % Green channel
blue = img(:,:,3); % Blue channel
%[m,n]=size(A);
D=padarray(A,[779 779],0,'post');
%D=padarray(A,[389 389],0,'both');
%h=padarray(D,[1,1],'post');


%N = size(red,1);
Yf = (fft2(red));
Y1f =(fft2(green));
Y2f = (fft2(blue));
Hf = fftshift(fft2(D));

% user interactive to enter the value of gamma
prompt='enter the value of gamma';
title='gamma value';
gamma=inputdlg(prompt,title);
gamma=str2double(gamma);

if abs(Hf)> gamma  % thresholding gamma
    iHf = 1./Hf;
else
    iHf = gamma;
end

% handle singular case (zero case)
%sHf = Hf.*(abs(Hf)>0)+1/gamma*(abs(Hf)==0);
%iHf = 1./sHf;

%lengthzero = length(abs(Hf)==0)
% invert Hf using threshold gamma
%iHf = iHf.*(abs(Hf)*gamma>1)+gamma*abs(sHf).*iHf.*(abs(sHf)*gamma<=1);
f_est = abs(ifft2(iHf.*Yf));
f_est1 = abs(ifft2(iHf.*Y1f));
f_est2 = abs(ifft2(iHf.*Y2f));
%actualImg = cat(3,channel1,channel2,channel3);
restoredimg = cat(3, f_est,f_est1,f_est2);
imshow(restoredimg,[]);

md = (originalimg - restoredimg).^2;
mdsize = size(md);
summation = 0;
for  i = 1:mdsize(1);
    for j = 1:mdsize(2);
        summation = summation + abs(md(i,j));
    end
end

erms=sqrt(summation);
psnr=20*log10(255/erms);
disp(['Calculated PSNR = ' num2str(psnr) ' dB']);


%image SSIM
img1=rgb2gray(originalimg);
img2=rgb2gray(restoredimg);

img1 = double(img1);
img2 = double(img2);
[M,N]=size(img1);
K = [0.01 0.03];
window = fspecial('gaussian', 11, 1.5);
L = 255;
f = max(1,round(min(M,N)/256));
%downsampling by f
%use a simple low-pass filter 
if(f>1)
    lpf = ones(f,f);
    lpf = lpf/sum(lpf(:));
    img1 = imfilter(img1,lpf,'symmetric','same');
    img2 = imfilter(img2,lpf,'symmetric','same');

    img1 = img1(1:f:end,1:f:end);
    img2 = img2(1:f:end,1:f:end);
end

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window = window/sum(sum(window));

mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;

if (C1 > 0 && C2 > 0)
   ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
else
   numerator1 = 2*mu1_mu2 + C1;
   numerator2 = 2*sigma12 + C2;
	denominator1 = mu1_sq + mu2_sq + C1;
   denominator2 = sigma1_sq + sigma2_sq + C2;
   ssim_map = ones(size(mu1));
   index = (denominator1.*denominator2 > 0);
   ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
   index = (denominator1 ~= 0) & (denominator2 == 0);
   ssim_map(index) = numerator1(index)./denominator1(index);
end

mssim = mean2(ssim_map);

return



