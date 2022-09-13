d=2;
q = sym('q',[d*3,1]);
assume(q,'real');

% a = q(1:d);
% b = q(d+1:2*d);
% p = q(2*d+1:3*d);
% 
% v = b - a;
% w = p - a;
% 
% c1 = dot(w,v);
% c2 = dot(v,v);
% assume(c2 > 0)
% c = c1 / c2;
% d = norm(p - (a + c*v));

d=2;
q = sym('q',[d*3,1]);
assume(q,'real');

p = q(1:d);
e0 = q(d+1:2*d);
e1 = q(2*d+1:3*d);

% 2D
e = e1 - e0;
num = e(2)*p(1) - e(1)*p(2) + e1(1)*e0(2) - e1(2)*e0(1);
d = sqrt(num * num / dot(e,e));

% 3D
% tmp = cross(e0-p,e1-p);
% e = e1 - e0;
% d = sqrt(dot(tmp,tmp) / dot(e,e));


g=gradient(d,q);
H=hessian(d,q);
% ccode(simplify(g))
% ccode(simplify(H))
matlabFunction(g, 'vars', {q},'File','gradient','Optimize',true)
matlabFunction(H, 'vars', {q},'File','hessian','Optimize',true)
%%%%%%%%%%%%%%
d=2;
q = sym('q',[d*2,1]);
assume(q,'real');

a = q(1:d);
b = q(d+1:2*d);
d = norm(a-b);


g=gradient(d,q);
H=hessian(d,q);
ccode(simplify(g))
ccode(simplify(H))
matlabFunction(g, 'vars', {q},'File','gradient','Optimize',true)
matlabFunction(H, 'vars', {q},'File','hessian','Optimize',true)


%%%%%%%%%%%%%%
syms d h k
psi = -k * (d-h)^2 * log(d/h);
%psi = (k/h/h)*(d-h)^2 * exp(-d/(h^2));
%psi = k*(d-h)^2;
dpsi = gradient(psi,d);
d2psi = hessian(psi,d);
ccode(psi)
ccode(dpsi)
ccode(d2psi)
