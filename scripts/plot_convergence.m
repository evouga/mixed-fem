ym = readDMAT('../output/ym_values.dmat');
a = readDMAT('../output/finalconvergence_converge_fem.dmat');
b = readDMAT('../output/finalconvergence_converge_mfem.dmat');
figure(2); clf;
loglog(ym,a);
hold on;
loglog(ym,b);
xlabel("Young's Modulus (Pa)");
ylabel("Final Gradient Norm");
legend("Vanilla FEM", "Ours");


figure(1); clf;

ym = readDMAT('../output/ym_values.dmat');
a = readDMAT('../output/convergence_grad_1.dmat');
b = readDMAT('../output/convergence_grad_2.dmat');
N = numel(ym);
M = size(a,2);

colormap(CustomColormap);
cmap = CustomColormap6; 
hold on;
for i=1:N
    semilogy(1:M,b(i,:),'--','Color',cmap(i,:));
    semilogy(1:M,a(i,:),'-','Color',cmap(i,:));
    if (i == 1)
        legend("MFEM", "Vanilla FEM");
    end
end
set(gca, 'YScale', 'log');
a= colorbar;
a.Label.String = 'Youngs Modulus (Pa)';
xlabel("Iterations");
ylabel("Newton Decrement");
% legend("Vanilla FEM", "MFEM");

% sqpf = fopen('conv_sqp.txt','r');
% wraf = fopen('conv_wrapd.txt','r');
% newton = fopen('conv_newton.txt','r');
% formatSpec = '%f';
% A = fscanf(sqpf,formatSpec);
% B = fscanf(wraf,formatSpec);
% C = fscanf(newton,formatSpec);
% 
% figure(1);clf; 
% loglog(1:18,A(1:18),'LineWidth',1); hold on;
% loglog(1:numel(B),B,'LineWidth',1);
% loglog(1:numel(C),C,'LineWidth',1);
% set(gca, 'YScale', 'log') % But you can explicitly force it to be logarithmic
% legend('MFEM','WRAPD','Vanilla FEM')
% xlabel('Iterations');
% ylabel('ND')
