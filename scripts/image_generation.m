clc
clearvars
close all

%------------------------------Real Space----------------------------------

Nx = 128;
Ny = 128;

x0_min = -1e-3;
x0_max =  1e-3;
y0_min = -1e-3;
y0_max =  1e-3;
x0 = linspace(x0_min,x0_max,Nx);
y0 = linspace(y0_min,y0_max,Ny);
[x,y] = meshgrid(x0,y0);

dx = x0(2) - x0(1);
dy = y0(2) - y0(1);

r = sqrt(x.^2+y.^2);

z2 = .5;
r2 = sqrt(x.^2+y.^2+z2^2);
% r2 = r2*0+1;

%-----------------------------Input Fields----------------------------------

lambda = 632.8e-9;
k = 2*pi/lambda;
E0          = 1;

for ii = 1:180000
        r1          = sqrt(rand(1)*80+x/50*(rand(1)-.5)+y/50*(rand(1)-.5));
        phi1        = k*r1;
        deviation   = rand(1,randi(3)) * 20;
        phi1        = phi1 + imresize(deviation, size(phi1));
        E1          = E0*exp(1i*phi1);
        phi2        = k*r2*(rand(1)-.5);
        E2          = E0*exp(1i*phi2);
        dp          = phi1 - phi2;
        dphi        = dp - min(dp(:));
        I           = abs(E1+E2).^2;
        save(['out' num2str(ii) '.mat'],"I","dphi");
        imagesc(I)
        colormap gray
        pause(.001)
end
