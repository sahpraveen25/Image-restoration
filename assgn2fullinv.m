A=imread('kernel.png');
%img=imread('Blurry1_1.jpg');
img= imread('Blurry2_1.jpg');
%img= imread('Blurry3_1.jpg');
red = img(:,:,1); % Red channel
green = img(:,:,2); % Green channel
blue = img(:,:,3); % Blue channel
%g=rgb2gray(g);
%[m,n]=size(A);
%D=padarray(A,[389 389],0,'both');
D=padarray(A,[779 779],0,'post');
%h=padarray(D,[1,1],'post');
%N = size(red,1);
Gg = (fft2(red));
Gg1 = (fft2(green));
Gg2 = (fft2(blue));

Hf =fftshift(fft2(D));
iHf =1./Hf;

Ffinv=Gg.*(iHf);
Ffinv1=Gg1.*(iHf);
Ffinv2=Gg2.*(iHf);

finv=(ifft2(Ffinv));
finv1=(ifft2(Ffinv1));
finv2=(ifft2(Ffinv2));
f_estim = cat(3, finv,finv1,finv2);
imshow(abs(f_estim));