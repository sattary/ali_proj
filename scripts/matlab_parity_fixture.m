% Fixed-parameter parity fixtures (not RNG matching).
Nx=128; Ny=128; x0=linspace(-1e-3,1e-3,Nx); y0=linspace(-1e-3,1e-3,Ny);
[x,y]=meshgrid(x0,y0); lambda=632.8e-9; k=2*pi/lambda; r2=sqrt(x.^2+y.^2+0.5^2);
cases={struct('u',0.21,'vx',0.13,'vy',-0.17,'a',-0.2,'dev',[1.5]), ...
       struct('u',0.61,'vx',-0.11,'vy',0.09,'a',0.1,'dev',[1.5 7.2]), ...
       struct('u',0.37,'vx',0.04,'vy',0.19,'a',0.3,'dev',[0.2 3.1 9.8])};
for i=1:numel(cases)
 c=cases{i}; r1=sqrt(c.u*80+x/50*c.vx+y/50*c.vy); phi1=k*r1+imresize(c.dev,[Ny Nx],'bicubic');
 phi2=k*r2*c.a; dp=phi1-phi2; dp_min=min(dp(:)); dphi=dp-dp_min; I=2+2*cos(dp);
 grad_phi2=cat(3,k*c.a*x./r2,k*c.a*y./r2);
 save(sprintf('matlab_fixture_%d.mat',i),'I','dphi','dp_min','grad_phi2','-v7');
end
