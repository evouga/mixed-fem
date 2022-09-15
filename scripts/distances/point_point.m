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