function fun_with_preconditioning
rng(1)

M = mmread('lhs_M.mkt');
K = mmread('lhs_K.mkt');
Gx = mmread('lhs_Gx.mkt');
Gs = mmread('lhs_Gs.mkt');
r_x = mmread('rhs_x.mkt');
r_gs = mmread('rhs_gs.mkt');
r_gl = mmread('rhs_gl.mkt');
b = [r_x; r_gs; r_gl];

Z = sparse(size(M,1), size(K,1));
Z2 = sparse(size(K,1), size(K,1));
A = [M Z Gx; Z' K Gs; Gx' Gs Z2];
% condest(A)
% cond(full(A))
k=16;
%[la, sa] = evals(A,k);
%[la, sa] = evals_diag(A,k);
%[la, sa] = evals_ilu(A,k);
%[la, sa] = evals_schur(A,M,K,Gx,Gs,k);
[P] = evals_saddle1(A,M,K,Gx,Gs,k);

% options.type = "ilutp";
% options.milu = "row";
% options.droptol = 1e-5;
% [L,U] = ilu(A,options);
% M = @(x) U \ (L \ x);

figure(1); clf;
[x,flag,relres,iter,resvec] = minres(A,b,1e-7, 100, P);
semilogy(resvec)

plot(la,'r.', 'MarkerSize', 15)
plot(sa,'g.', 'MarkerSize', 15)
hold off
legend('Largest magnitude','Smallest magnitude')
xlabel('Real axis')
ylabel('Imaginary axis')
end

function [la,sa]=evals(A,k)
condest(A)
la = eigs(A,k,'largestabs');
sa = eigs(A,k,'smallestabs');
end

function [la,sa]=evals_diag(A,k)
P = sparse(diag(1./sqrt(sum(abs(A).^2))));
[L,U] = lu(A);
Afun = @(x) P * A * x;
Afuninv = @(x) U \ (L \ (P \ x));
la = eigs(Afun,size(P,1),k,'largestabs');
sa = eigs(Afuninv,size(P,1),k,'smallestabs');
condest(P * A)
end

function [la,sa]=evals_ilu(A,k)
options.type = "ilutp";
options.milu = "row";
options.droptol = 1e-5;
tic
[L,U] = ilu(A,options);
toc
diff=norm(A - L*U,'fro')
% M    = (LU)^-1 A = U^-1 L^-1 A
% M^-1 = A^-1 L * U
Afun = @(x) U \ (L \ (A*x));
Afuninv = @(x) A \ (L * (U *x));
la = eigs(Afun,size(A,1),k,'largestreal');
sa = eigs(Afuninv,size(A,1),k,'smallestreal');
la=real(la);
sa=real(sa);
condest(U \ (L \ (A)))
end

function [la,sa]=evals_schur(A,M,K,Gx,Gs,k) 
    Mlump = sparse(1:size(M,1), 1:size(M,1), sum(M,2));
    Mlumpinv = sparse(1:size(M,1), 1:size(M,1), 1./sum(M,2));
    
    Minv = inv(M);
    Kinv = inv(K);
    S1 = M;
    S2 = K;
    
    R = ones(size(Gx,2),1) * 1e0;
    R = sparse(1:size(R,1), 1:size(R,1), R);
    S3 = Gx'*Mlumpinv*Gx + R;
    n1 = size(S1,1);
    n2 = size(S2,1);
    n3 = size(S3,1);
    Z = sparse(size(M,1), size(K,1));
    Z2 = sparse(size(K,1), size(K,1));
    Pinv = [Minv Z Z; ...
            Z' Kinv Z2; ...
            Z' Z2 inv(S3)];

    function z=Afun(x)
        z= Pinv*A*x;
%         z1 = S1 \ z(1:n1);
%         z2 = S2 \ z(n1+1:n1+n2);
%         z3 = S3 \ z(n1+n2+1:n1+n2+n3);
%         z =[z1;z2;z3];
    end
    Afuninv = @(x) A \ (Pinv \ x);
    la = eigs(@Afun,size(A,1),k,'largestabs');
    sa = eigs(Afuninv,size(A,1),k,'smallestabs');
    spy(Pinv)
    condest(Pinv*A)
end

function [P]=evals_saddle1(A,M,K,Gx,Gs,k)
    
    % Lumped mass matrix
    Mlump = sparse(1:size(M,1), 1:size(M,1), sum(M,2));
    Mlumpinv = sparse(1:size(M,1), 1:size(M,1), 1./sum(M,2));
    
    % Inverse of hessians
    Minv = inv(M);
    Kinv = inv(K);
    
    % Regularization
    R = ones(size(Gx,2),1) * 1e-8;
    R = sparse(1:size(R,1), 1:size(R,1), R);

    % Top left block of 2x2 schur complement preconditioner
    Z = sparse(size(M,1), size(K,1));
    A1 = [M Z; Z' K];
    A1inv = [Minv Z; Z' Kinv];

    % Off diagonal block
    B = [Gx; Gs];
    %S = Gs * Gx' * Mlumpinv * Gx * Gs + R;
    S = B' * A1inv * B;

    % 2x2 schur complement preconditioner and its inverse
    Z1 = sparse(size(S,1), size(A1,1));
    P = [A1 Z1'; Z1 S];
asdf=min(abs(eig(full(K))))
    % 3x3 block diagonal preconditioner
    Z1 = sparse(size(M,1), size(K,1));
    Z2 = sparse(size(K,1), size(K,1));
    S1 = Gx' * Mlumpinv * Gx + R;
    S2 = Gs * inv(S1) * Gs + K;
    S2 = K;
    S2 = sparse(1:size(K,1), 1:size(K,1), sum(K,2));


%     tmp = inv(Gs) * S1 * inv(Gs);
%     S2inv = tmp - tmp*Kinv*tmp;
%     norm(S2inv-inv(S2),'fro')
% 
%     S2inv = sappinv(S2,S1);
%     spy(S2inv)
%     norm(S2inv-inv(S2),'fro')
% 
    P = [ M  Z1 Z1; ...
         Z1' S2 Z2; ...
         Z1' Z2 S1];
   
    % 1. Fixed Corot
    % 2. Try out preconditioner with different stiffnesses / deformations
    % 3. Vary tolerance of inner solver and see how that affects outer
    % convergence
    %
    Pinv = inv(P);
%     Pinv = [ Minv  Z1 Z1; ...
%              Z1' S2inv Z2; ...
%              Z1' Z2 inv(S1)];

    [~,D] = eig(full(Pinv*A),'vector');
    [~,D0] = eig(full(A),'vector');
    D = sort(abs(real(D)));
    D0 = sort(abs(real(D0)));
    s1 = max(D) / min(D);
    s2 = max(D0) / min(D0);
    fprintf("Original condition number: %.5g,\nNew: %.5g\n",s2,s1);
    clf;
    plot(D,'.-');
    hold on;
    plot(D0,'.-');
end
