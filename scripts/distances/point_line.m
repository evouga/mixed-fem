d=3;
q = sym('q',[d*3,1]);
assume(q,'real');

p = q(1:d);
e0 = q(d+1:2*d);
e1 = q(2*d+1:3*d);

% 2D
% e = e1 - e0;
% num = e(2)*p(1) - e(1)*p(2) + e1(1)*e0(2) - e1(2)*e0(1);
% d = sqrt(num * num / dot(e,e));

% 3D
tmp = cross(e0-p,e1-p);
e = e1 - e0;
d = sqrt(dot(tmp,tmp) / dot(e,e));


g=gradient(d,q);
H=hessian(d,q);
% ccode(simplify(g))
% ccode(simplify(H))
matlabFunction(g, 'vars', {q},'File','gradient','Optimize',true)
matlabFunction(H, 'vars', {q},'File','hessian','Optimize',true)