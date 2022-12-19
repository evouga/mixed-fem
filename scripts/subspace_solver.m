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
b = [r_x; r_gs; r_gl];
%%

%% Creating projection matrices
Mlump = sparse(1:size(M,1), 1:size(M,1), sum(M,2));
Msqrt = sqrt(Mlump);
Msqrtinv = sparse(1:size(M,1), 1:size(M,1), 1./sqrt(sum(M,2)));
Minv = sparse(1:size(M,1), 1:size(M,1), 1./sum(M,2));
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

% (BM^{-1}B' + D^2)la = BM^{-1}gx + CH^{-1}gs - c
RHS3 = Gx'* (M\r_x) + D*fy - c;
la3 = LHS2\RHS3;
x3 = M \(r_x - Gx*la3);
y3 = -D*la3 + fy;
s3 = Hsqrtinv * y3;

% x, y = H^{1/2}s
% x'Mx + ||y||^2 - x'gx - y'fy
% fy = H^{-1/2}gs
% Mx - gx + la'B
% y - fy + la'D
% Bx + Dy - c = 0
% B(M^{-1}(gx - B'la)) + Dy = c
% -B M^-1 B'la + Dy = c - BM^{-1}gx
% -B M^-1 B'la + D(fy - la'D) = c - BM^{-1}gx
% (-B M^-1 B - DD)la = c - BM^{-1}gx - CH^{-1}gs
% (BM^{-1}B' + D^2)la = BM^{-1}gx + CH^{-1}gs - c

% Mx - gx + la'B
% Hs - gs + la'C
% Bx + Dy - c
% B(M-1(gx-B'la)) + CH^{-1}(gs-C'la)
%LHS2 = G*G' + D*D;
% la4 = (Gx'*Minv*Gx + Gs*inv(K)*Gs)\-(c - Gx'*Minv*r_x - Gs*inv(K)*r_gs);
% x4 = Minv*(r_x-Gx*la4);
% s4 = inv(K)*(r_gs-Gs*la4);
% norm(la4-la)
% norm(la3-la4)
%cond(full(LHS))
%cond(full(LHS2))
%cond(full(A))
r1 = A * [x;s;la] - b;
r2 = A * [x2;s2;la2] - b;
% r3 = A * [x4;s4;la4] - b;

norm(r1) / norm(b)
norm(r2) / norm(b)
% norm(r3) / norm(b)
[W,D] = eig(full(A));
norm(W*D*W' - A,'fro')
k=1758;
B=W(:,end-k+1:end);
norm(B*D(end-k+1:end,end-k+1:end)*B' - A,'fro')

z=(B'*B)\(B'*r2);
plot(1:k,z);
D(1:5,1:5)


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
