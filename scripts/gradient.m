function g = gradient(in1)
%GRADIENT
%    G = GRADIENT(IN1)

%    This function was generated by the Symbolic Math Toolbox version 8.7.
%    16-Aug-2022 20:26:13

q1 = in1(1,:);
q2 = in1(2,:);
q3 = in1(3,:);
q4 = in1(4,:);
t2 = -q3;
t3 = -q4;
t4 = q1+t2;
t5 = q2+t3;
t6 = abs(t4);
t7 = abs(t5);
t8 = sign(t4);
t9 = sign(t5);
t10 = t6.^2;
t11 = t7.^2;
t12 = t10+t11;
t13 = 1.0./sqrt(t12);
t14 = t6.*t8.*t13;
t15 = t7.*t9.*t13;
g = [t14;t15;-t14;-t15];