d=3;
R=sym('R',[d d]);
S = sym('S',[6,1]);
% S = sym('S',[d,d]);
L = sym('L',[9,1]);
syms mu la
assume(R,'real')
assume(S,'real')
assume(mu,'real')
assume(L,'real')

F = [S(1) S(6) S(5);
     S(6) S(2) S(4);
     S(5) S(4) S(3)];
 F = [S(1) S(4) S(5);
     S(4) S(2) S(6);
     S(5) S(6) S(3)];
 %F=S;
 %R=eye(3);
 t=13;
%  R = [cos(t) -sin(t) 0; sin(t) cos(t) 0; 0 0 1]
R'*R
W = [                          
  [R(1,1) 0 0 0 R(1,3) R(1,2)]
  [0 R(1,2) 0 R(1,3) 0 R(1,1)]
  [0 0 R(1,3) R(1,2) R(1,1) 0]
  [R(2,1) 0 0 0 R(2,3) R(2,2)]
  [0 R(2,2) 0 R(2,3) 0 R(2,1)]
  [0 0 R(2,3) R(2,2) R(2,1) 0]
  [R(3,1) 0 0 0 R(3,3) R(3,2)]
  [0 R(3,2) 0 R(3,3) 0 R(3,1)]
  [0 0 R(3,3) R(3,2) R(3,1) 0]
  ];

% W = [                          
%   [R(1,1) 0 0 0 R(1,3) R(1,2)]
%   [R(2,1) 0 0 0 R(2,3) R(2,2)]
%   [R(3,1) 0 0 0 R(3,3) R(3,2)]
%   [0 R(1,2) 0 R(1,3) 0 R(1,1)]
%   [0 R(2,2) 0 R(2,3) 0 R(2,1)]
%   [0 R(3,2) 0 R(3,3) 0 R(3,1)]
%   [0 0 R(1,3) R(1,2) R(1,1) 0]
%   [0 0 R(2,3) R(2,2) R(2,1) 0]
%   [0 0 R(3,3) R(3,2) R(3,1) 0]];

% % stable neohookean
%F=R*F; (determinant of R=1, F'*F = S'S)
I3=det(F);
I2=trace(F'*F);
snh= 0.5*mu*(I2-d)- mu*(I3-1)+ 0.5*la*(I3-1)^2;
H=simplify(hessian(snh,S(:)));
g=simplify(gradient(snh,S(:)));
ccode(snh)
ccode(H)
ccode(g)

% neohookean
%F=R*S;
%J=det(F);
%I3=trace(F'*F)/J^(2/3);
%snh=0.5*mu*(I3-3)+ 0.5*la*(J-1)^2;

% Corotational material model
arap= mu*0.5*trace( (F - eye(d))*(F - eye(d))');
corot=  1*la*0.5*trace(F-eye(d))^2 + 2*arap;
H=hessian(corot,S(:));
g=gradient(corot,S(:));
%ccode(H)
%ccode(g)

Hinv = inv(H);
%Hinv = sym('Hinv',[6,6]);
%assume(Hinv,'real');

WHW=W*Hinv*W';

% Wst - WH^-1g
Ws_Hinvg = W*(S-Hinv*g);

% -H^-1*g + H^-1*W'*lambda
ds = Hinv*(W'*L - g);

%ccode(Hinv)

ccode(WHW)
ccode(Ws_Hinvg)
ccode(ds)