d=2;
q = sym('q',[d*3,1]);
assume(q,'real');

a = q(1:d);
b = q(d+1:2*d);
p = q(2*d+1:3*d);

v = b - a;
w = p - a;

c1 = dot(w,v);
c2 = dot(v,v);
assume(c2 > 0)
c = c1 / c2;
d = norm(p - (a + c*v));

g=gradient(d,q);
H=hessian(d,q);
%ccode(simplify(g))
%ccode(simplify(H))
%matlabFunction(g, 'vars', {q},'File','gradient','Optimize',true)
%matlabFunction(H, 'vars', {q},'File','hessian','Optimize',true)
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
