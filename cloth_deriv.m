L = sym('L',[9 1]);
q = sym('q',[9 1]);
N = sym('N',[3 2]);
assume(q,'real')
assume(N,'real')

dx1 = [-eye(3) eye(3) 0*eye(3)]*q;
dx2 = [-eye(3) 0*eye(3) eye(3)]*q;

n = cross(dx1,dx2);
Px = dx1 / norm(dx1);
Py = cross(dx2,n);
Py = Py / norm(Py);
P = [Px'; Py'];
x = [q(1:3) q(4:6) q(7:9)];
F = P*x*N;

J = jacobian(F(:), q);
ccode(J)
ccode(simplify(J))
ccode(F(:))
H=simplify(hessian(snh,sval(:)));
g=simplify(gradient(snh,sval(:)));
ccode(snh)
ccode(H)
ccode(g)
