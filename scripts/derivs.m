d=2;
F=sym('F',[d d]);
s = sym('S',[3,1]);
sval = sym('sval',[3,1]);
syms mu la
assume(F,'real')
assume(s,'real')
assume(mu,'real')
assume(la,'real')
assume(sval,'real')
% 
% S = [s(1) s(4) s(5);
%      s(4) s(2) s(6);
%      s(5) s(6) s(3)];
S = [s(1) s(3);
     s(3) s(2)];

% stable neohookean
I3=det(F);
I2=trace(F'*F);
% I3=det(S);
% I2=trace(S'*S);
% I2 = sum(sval.^2);
% I3 = prod(sval);


snh= 0.5*mu*(I2-d)- mu*(I3-1)+ 0.5*la*(I3-1)^2;
H=simplify(hessian(snh,F(:)));
g=simplify(gradient(snh,F(:)));
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
