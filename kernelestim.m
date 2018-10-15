%read image
img=imread('Blurry1_1.jpg');
A=rgb2gray(img);
I=double(A)/255; %for now, assume we know I

b=I; % no convolution
m=3; %size of the (square) kernel

%or actually blur b.
%K=[0 0 0; 0 1 0; 0 0 0]; K=K./sum(sum(k)); m=max(size(K));
%b=conv2(b,K);

%init
m2=(m+1)/2;
[r c,dep] = size(I);
B=b(:); %gives column major vectorization of blurred image b.

%noise may be added to I here, to simulate blurred-noisy pair.
%I=max(0,I+(rand(size(I))-0.5)*.02); % 1-percent uniform random error

%pad I so the out-of-bounds error doesn't occur
Ip=zeros(r+m-1,c+m-1); %assuming single color channel
Ip(m2+(0:r-1), m2+(0:c-1)) = I;

A=zeros(r*c,m*m);
%column major to stay consistent with b
k=0;
for j=1:c
    disp(j);
    pause(.1);
    for i=1:r 
        k=k+1;
        %cut out a portion of Ip and vectorize it
        Ak=Ip(i+(0:m-1), j+(0:m-1) );
        A(k,:)=Ak(:)';
    end
end

%solve...
x=A\B; %<-- not a good idea. the paper has a better approach ;-)

%transform into a matrix
blur_kernel =reshape(x,m,m)