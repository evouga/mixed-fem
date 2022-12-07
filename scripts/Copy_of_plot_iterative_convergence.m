a = readDMAT('../output/finalconvergence_converge.dmat');
figure(1); clf;
loglog(1:numel(a),a,'.-','MarkerSize',15);
xlabel("tolerance");
ylabel("Gradient Norm");
title("Convergence after 10 iterations")


figure(2); clf;
a = readDMAT('../output/convergence.dmat');
N = size(a,1);
M = size(a,2);

%colormap(CustomColormap);
colormap(winter(1000))
cmap = winter(N); 
hold on;
for i=1:N
    semilogy(1:M,a(i,1:M),'-','LineWidth',2);%,'Color',cmap(i,:));

end
legend("Primal Condensation (CG + ICHOL)", ...
    "Indefinite System (Minres + Block Preconditioner)", ...
    "Dual Condensation (CG + Diag Preconditioner)",...
    "Indefinite System (ADMM solver)");
    
set(gca, 'YScale', 'log');
%a= colorbar;
%a.Label.String = "tolerances";
xlabel("Newton iteration");
ylabel("Gradient Norm");
%title("Convergence of Collision Simulation FEM vs MFEM")
