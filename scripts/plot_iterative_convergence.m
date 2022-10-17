tol = readDMAT('../output/ym_values.dmat');
a = readDMAT('../output/finalconvergence_converge.dmat');
figure(1); clf;
loglog(tol,a,'.-','MarkerSize',15);
xlabel("tolerance");
ylabel("Gradient Norm");
title("Convergence after 10 iterations")


figure(2); clf;
a = readDMAT('../output/convergence.dmat');
N = numel(tol);
M = size(a,2);

%colormap(CustomColormap);
colormap(winter(1000))
cmap = winter(N); 
hold on;
for i=1:N
    semilogy(1:M,a(i,1:M),'-','Color',cmap(i,:));
end
set(gca, 'YScale', 'log');
%a= colorbar;
%a.Label.String = "tolerances";
xlabel("Iterations");
ylabel("Gradient Norm");
%title("Convergence of Collision Simulation FEM vs MFEM")
