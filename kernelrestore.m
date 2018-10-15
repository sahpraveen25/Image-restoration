p=[0 -0 0; -0 1 0; 0 0 -0];
img=imread('Blurry1_1.jpg');
%originalimg=double(imread('Groundtruth1_1_1.jpg'));
%g=rgb2gray(g);
red = img(:,:,1); % Red channel
green = img(:,:,2); % Green channel
blue = img(:,:,3); % Blue channel
%[m,n]=size(A);
D=padarray(A,[779 779],0,'post');
%D=padarray(A,[389 389],0,'both');
%=padarray(D,[1,1],'post');


N = size(red,1);
Gg = (fft2(red));
Gg1 = (fft2(green));
Gg2 = (fft2(blue));
Hh = fftshift(fft2(p,N,N));
%user interactive to enter the value of gamma
prompt='enter the value of k';
title='k value';
K=inputdlg(prompt,title);
K=str2double(K);
cH = conj(Hh);
HcH = Hh .* cH;
K = K * max(max(abs(HcH)));
w = cH ./ (HcH + K);
Ffwin=w.*Gg;
Ffwin1=w.*Gg1;
Ffwin2=w.*Gg2;

%H2=abs(Hh).^2; 
%Ffwin=H2.*Gg./((H2+K).*Hh);
%Ffwin1=H2.*Gg1./((H2+K).*Hh);
%Ffwin2=H2.*Gg2./((H2+K).*Hh);

fwin=abs(ifft2(Ffwin));
fwin1=abs(ifft2(Ffwin1));
fwin2=abs(ifft2(Ffwin2));
restoredimg = cat(3, fwin,fwin1,fwin2);
%subplot(2,1,1)
imshow(restoredimg,[]);