d=3;
q = sym('q',[d*4,1]);
assume(q,'real');

p = q(1:d);
t0 = q(d+1:2*d);
t1 = q(2*d+1:3*d);
t2 = q(3*d+1:4*d);

normal = cross(t1-t0, t2-t0);
point_to_plane = dot(p-t0, normal);
d = sqrt(point_to_plane*point_to_plane/dot(normal,normal));

g=gradient(d,q);
H=hessian(d,q);
% ccode(simplify(g))
% ccode(simplify(H))
matlabFunction(g, 'vars', {q},'File','gradient','Optimize',true)
%matlabFunction(H, 'vars', {q},'File','hessian','Optimize',true)