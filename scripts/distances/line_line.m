d=3;
q = sym('q',[d*4,1]);
assume(q,'real');

a0 = q(1:d);
a1 = q(d+1:2*d);
b0 = q(2*d+1:3*d);
b1 = q(3*d+1:4*d);

normal = cross(a1-a0, b1-b0);
line_to_line = dot(b0-a0, normal);
d = sqrt(line_to_line * line_to_line / dot(normal, normal));

g=gradient(d,q);
H=hessian(d,q);
% ccode(simplify(g))
% ccode(simplify(H))
matlabFunction(g, 'vars', {q},'File','gradient','Optimize',true)
%matlabFunction(H, 'vars', {q},'File','hessian','Optimize',true)