function subspace_solver
rng(1)

%% Reading in matrices
M = mmread('lhs_M.mkt');
K = mmread('lhs_K.mkt');
Gx = mmread('lhs_Gx.mkt');
Gs = mmread('lhs_Gs.mkt');
r_x = mmread('rhs_x.mkt');
r_gs = mmread('rhs_gs.mkt');
r_gl = mmread('rhs_gl.mkt');
<<<<<<< HEAD
=======
r_subspace = mmread('rhs_sub.mkt');
>>>>>>> 1ba56d32a28af00c4b56bd4700758b0df4217e07
b = [r_x; r_gs; r_gl];
%%

%% Creating projection matrices
Mlump = sparse(1:size(M,1), 1:size(M,1), sum(M,2));
Msqrt = sqrt(Mlump);
Msqrtinv = sparse(1:size(M,1), 1:size(M,1), 1./sqrt(sum(M,2)));
Hsqrt = K;
Hsqrtinv = K;

L = 3;
N = size(K,1) / L;
for i=1:N
    K_i = full(K(L*(i-1)+1:L*i,L*(i-1)+1:L*i));
    [V,D] = eig(K_i);
    K_i_sqrt = V * sqrt(D) * V';
    Hsqrt(L*(i-1)+1:L*i,L*(i-1)+1:L*i) = K_i_sqrt;
    Hsqrtinv(L*(i-1)+1:L*i,L*(i-1)+1:L*i) = inv(K_i_sqrt);
    %test = K_i_sqrt * K_i_sqrt;
end
%%

Z = sparse(size(M,1), size(K,1));
Z2 = sparse(size(K,1), size(K,1));
A = [M Z Gx; Z' K Gs; Gx' Gs Z2];

G = Gx' * Msqrtinv;
D = Gs * Hsqrtinv; 
Dinv = inv(Gs) * Hsqrt;

fp = Msqrtinv * r_x;
fy = Hsqrtinv * r_gs;
c = r_gl;

LHS = G*G'*Dinv + D;
RHS = c + G*G'*Dinv*fy - G*fp;
y = LHS\RHS;
la = -Dinv * (y - fy);
p = -G'*la + fp;

LHS2 = G*G' + D*D;
RHS2 = G*fp + D*fy - c;

la2 = LHS2\RHS2;
p2 = -G'*la2 + fp;
y2 = -D*la2 + fy;

x = Msqrtinv * p;
s = Hsqrtinv * y;
x2 = Msqrtinv * p2;
s2 = Hsqrtinv * y2;
%cond(full(LHS))
%cond(full(LHS2))
%cond(full(A))
norm(A * [x;s;la] - b) / norm(b)
norm(A * [x2;s2;la2] - b) / norm(b)
%norm(G*p+D*y-c)

<<<<<<< HEAD
% r_sub = mmread('rhs_sub.mkt');
% x_sub = mmread('x_sub.mkt');
% s_sub = mmread('s_sub.mkt');
% l_sub = mmread('la_sub.mkt');
% y_sub = mmread('y_sub.mkt');
% p_sub = mmread('p_sub.mkt');
% rhs2 = mmread('rhs2.mkt');
% lhs2 = mmread('lhs2.mkt');
=======
r_sub = mmread('rhs_sub.mkt');
x_sub = mmread('x_sub.mkt');
s_sub = mmread('s_sub.mkt');
l_sub = mmread('la_sub.mkt');
y_sub = mmread('y_sub.mkt');
p_sub = mmread('p_sub.mkt');
rhs2 = mmread('rhs2.mkt');
lhs2 = mmread('lhs2.mkt');
>>>>>>> 1ba56d32a28af00c4b56bd4700758b0df4217e07
% Gl_sub = mmread('Gla_sub.mkt');
% lump = mmread('lump_sub.mkt');
% options.type = "ilutp";
% options.milu = "row";
% options.droptol = 1e-5;
% [L,U] = ilu(A,options);
% M = @(x) U \ (L \ x);
% P = sparse(diag(sum(abs(A))));
% P = sparse(eye(size(A)));
[~,D] = eig(full(LHS),'vector');
[~,D0] = eig(full(LHS2),'vector');
%[~,D0] = eig(full(A),'vector');
D = sort(abs(real(D)));
D0 = sort(abs(real(D0)));
s1 = max(D) / min(D);
s2 = max(D0) / min(D0);
fprintf("Original condition number: %.5g,\nNew: %.5g\n",s2,s1);
figure(2); clf;
plot(D,'.-');
hold on;
plot(D0,'.-');

figure(1); clf;
%[x,flag,relres,iter,resvec] = minres(A,b,1e-7, 500, P);
[x,flag,relres,iter,resvec] = minres(LHS,RHS,1e-7, 200);
[x,flag,relres,iter,resvec2] = minres(LHS2,RHS2,1e-7, 200);
% [x,flag,relres,iter,resvec] = gmres(A,b,10,1e-7, 100, P);
semilogy(resvec);
hold on;
semilogy(resvec2)
end
