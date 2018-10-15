A=imread('kernel.png'); % image loading
originalimg=imread('Groundtruth4_1_1.jpg'); % groundtruth data
img=imread('Blurry4_1.jpg');
red = img(:,:,1); % Red channel
green = img(:,:,2); % Green channel
blue = img(:,:,3); % Blue channel
%g=rgb2gray(g);
%figure(1);
%imshow(g);

%[m,n]=size(A);
D=padarray(A,[779 779],0,'post'); % padding zero to the left and bottom
%D=padarray(A,[389 389],0,'both');
%h=padarray(D,[1,1],'post');


%N = size(red,1); 
Gg = (fft2(red)); %frequncy domain
Gg1 = (fft2(green));
Gg2 = (fft2(blue));
Hh = fftshift(fft2(D));

H2=abs(Hh).^2; %Taking magnitude
% Constrained least square method using smoothing operator
% p(x,y) = [0  -1   0]
%          [-1  4  -1]
%          [ 0 -1   0]
prompt='enter the value of gamma';
title='gamma value';
gamma=inputdlg(prompt,title);
gamma=str2double(gamma);
p=[0 -1 0; -1 4 -1; 0 -1 0];  % mask 
Pp=fft2(p,N,N); % taking magnitude
Hhcls=conj(Hh).*Gg./(H2+gamma*abs(Pp).^2); % applying lsf
Hhcls1=conj(Hh).*Gg1./(H2+gamma*abs(Pp).^2);
Hhcls2=conj(Hh).*Gg2./(H2+gamma*abs(Pp).^2);
hcls=abs(ifft2(Hhcls)); % convertinging to spatial domain
hcls1=abs(ifft2(Hhcls1));
hcls2=abs(ifft2(Hhcls2));
restoredimg = cat(3, hcls,hcls1,hcls2); % concatenate RGB for dispaly
imshow(restoredimg, []);
restoredimg=rgb2gray(restoredimg);

%ssimval = ssim(restoredimg,originalimg);
%fprintf('The SSIM value is %0.4f.\n',ssimval);
% now start iteration
[m,n,dep]=size(originalimg);
mn=m*n;
sigma = 256*10^(-46/20); %calculating sigma from psnr
eta2=mn*sigma^2;  % (5.9-12), m_eta = 0 in this example
disp(['||eta||^2 = ' num2str(eta2)])
icnt=0; cntmax=20;
converge=0; 
accuracy=mn*sigma;  % accuracy factor a in (5.9-8), every pixel within sigma
gub=10; glb=0; % upper and lower bounds of gamma initially
while converge==0, % start iteration
   icnt=icnt+1; disp(['iteration # ' int2str(icnt)]);
   % Use Parsevel theorem, ||r||^2 = sum sum |R|^2
   Rr=abs(Gg-Hh.*Hhcls);
   r2=(sum(sum(abs(Rr).^2))-Rr(1,1)^2)/mn;  % remove the mean value
   disp(['||r||^2 = ' num2str(r2) ', ||eta||^2 = ' num2str(eta2)])
   if abs(r2-eta2)<accuracy, converge=1;  %done, do nothing
   else
      if r2 < eta2, % gamma too small
         if gamma > glb, glb=gamma;
         end % update lower bound
         % gamma=gamma*(1+0.9^icnt); disp(['gamma increased to: ' num2str(gamma)]);
      elseif r2 > eta2, % gamma too large
         if gamma < gub, gub=gamma;
         end % update upper bound
         % gamma=gamma*(1-0.9^icnt); disp(['gamma decreased to: ' num2str(gamma)]);
      end
      gamma=0.5*(gub+glb);  % use bisection search
      Hhcls=conj(Hh).*Gg./(H2+gamma*abs(Pp).^2);
      Hhcls1=conj(Hh).*Gg./(H2+gamma*abs(Pp).^2);
      Hhcls2=conj(Hh).*Gg./(H2+gamma*abs(Pp).^2);
      hcls1=abs(ifft2(Hhcls));
      hcls2=abs(ifft2(Hhcls1));
      hcls3=abs(ifft2(Hhcls2));
      hclss = cat(3, hcls1,hcls2,hcls3);
     
      if rem(icnt-1,3)==0, % plot 1, 4, 7, 10, 13 iterations
        figure(2),
        eval(['subplot(23' int2str(ceil(icnt/3)) '),'])
        imagesc(hclss);
        %colormap('gray')
        disp(['gamma = ' num2str(gamma) ', enter any key to continue ...'])
      end
      pause
      if icnt == cntmax, 
         converge=1; disp('Reach max. iteration count, stop!')
      end
   end % if abs(... loop
end % while loop
imagesc(hclss,[]),title(['iteration ' int2str(icnt) ', gamma=' num2str(gamma)])


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