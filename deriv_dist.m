d=2;
q = sym('q',[d*3,1]);
assume(q,'real');

a = q(1:d);
b = q(d+1:2*d);
p = q(2*d+1:3*d);

e = b-a;
normal = e;
normal(1) = e(2);
normal(2) = -e(1);
normal = normal / norm(normal);
d = dot(p-a, normal);
g=gradient(d,q);
ccode(g)

syms d h k
psi = -k * (d-h)^2 * log(d/h);
%psi = (k/h/h)*(d-h)^2 * exp(-d/(h^2));
%psi = k*(d-h)^2;
dpsi = gradient(psi,d);
d2psi = hessian(psi,d);
ccode(psi)
ccode(dpsi)
ccode(d2psi)
