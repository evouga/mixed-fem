d=3;
R=sym('R',[d d]);
F=sym('F',[3 3]);
s = sym('S',[3,1]);
sval = sym('sval',[3,1]);
L = sym('L',[9,1]);
syms mu la
assume(R,'real')
assume(F,'real')
assume(s,'real')
assume(mu,'real')
assume(L,'real')
assume(sval,'real')

S = [s(1) 0 s(3);
     0    1 0;
     s(3) 0 s(2)];
d=size(S,1);

% stable neohookean
I3=det(F);
I2=trace(F'*F);
I3=det(S);
I2=trace(S'*S);


snh= 0.5*mu*(I2-d)- mu*(I3-1)+ 0.5*la*(I3-1)^2;
H=simplify(hessian(snh,s(:)));
g=simplify(gradient(snh,s(:)));
ccode(snh)
ccode(H)
ccode(g)

% neohookean
%nh = 0.5*mu*(I2/(I3^(2/3)) - 3) + 0.5*la*(I3-1)^2;
nh = 0.5*mu*(I2- 3) - mu*log(I3) + 0.5*la*(log(I3))^2;
% H=(hessian(nh,s(:)));
% g=gradient(nh,s(:));
H=(hessian(nh,F(:)));
g=gradient(nh,F(:));
ccode(nh)
ccode(H)
ccode(g)

%F=R*S;
%J=det(F);
%I3=trace(F'*F)/J^(2/3);
%snh=0.5*mu*(I3-3)+ 0.5*la*(J-1)^2;

% Corotational material model
arap= mu*0.5*trace( (S - eye(d))*(S - eye(d))');
H=hessian(arap,s(:));
g=gradient(arap,s(:));
ccode(arap)
ccode(H)
ccode(g)

corot = la*0.5*trace(S-eye(d))^2 + 2*arap;
H=hessian(corot,s(:));
g=gradient(corot,s(:));
ccode(corot)
ccode((H))
ccode(g)
