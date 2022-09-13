ym = readDMAT('../output/dhat_values.dmat');
a1 = readDMAT('../output/finalconvergence_converge_fem_1e4.dmat');
a2 = readDMAT('../output/finalconvergence_converge_fem_1e5.dmat');
a3 = readDMAT('../output/finalconvergence_converge_fem_1e6.dmat');
b = readDMAT('../output/finalconvergence_converge_mfem.dmat');
figure(2); clf;
loglog(ym,a1);
hold on;
loglog(ym,a2);
loglog(ym,a3);
loglog(ym,b);
xlabel("dhat");
ylabel("||\delta x||");
legend("Vanilla IPC (\kappa=1e4)","Vanilla IPC (\kappa=1e5)", "Vanilla IPC (\kappa=1e6)", "Ours");
title("Convergence after 50 iterations")


figure(1); clf;

a = readDMAT('../output/convergence_fem_1e6.dmat');
b = readDMAT('../output/convergence_mfem.dmat');
N = numel(ym);
M = size(a,2);

%colormap(CustomColormap);
colormap(winter(1000))
cmap = winter(N); 
hold on;
for i=1:N
    semilogy(1:M,a(i,:),'-','Color',cmap(i,:));
    semilogy(1:M,b(i,:),'--','Color',cmap(i,:));

    if (i == 1)
        legend("Vanilla IPC", "Ours");
    end
end
set(gca, 'YScale', 'log');
a= colorbar;
a.Label.String = "\hat d";
xlabel("Iterations");
ylabel("||\delta x||");
title("Convergence of Collision Simulation FEM vs MFEM")
