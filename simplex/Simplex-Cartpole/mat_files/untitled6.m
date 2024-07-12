% clear, clc

% x_set = [-0.9, 0.9];
% theta_set = [-0.8, 0.8];
% 
% A = [1,    0.03333333,    0,            0;
%      0,    1,             -0.05649123,  0;
%      0,    0,             1,            0.03333333;
%      0,    0,             0.89802632,   1        ];
% 
% B = [0; 0.03341688; 0; -0.0783208];
% 
% D = [1/0.9,   0,        0,        0;
%        0,     0,       1/0.8,     0;
%        0,     1/100,    0,         0;
%        0,     0,       0,     1/100];
% 
% 
% n = 4;
% alpha = 0.865;
% beta = 0.002;
% invV = 1/beta;
% c = 1/15;
% 
% cvx_begin sdp
% 
%     variable Q(n,n) symmetric;
%     variable R(1,4);
% 
%     minimize -log_det(Q)
% 
%     subject to
% %         Q > 0;
%         D * Q * D' - eye(4) < 0
% %         [alpha*Q,      Q*A' + R'*B';
% %           A*Q + B*R,  Q] > 0
%         Z = A+B*R;
% %         Q*Z' + Z*Q < zeros(4)
% 
% %     [Q,      R';
% %      R,      1/beta] > 0
% 
% cvx_end
% 
% P = inv(Q)
% F = R*P
% A + B*F
% 
% C = eig((A+B*F)'*P*(A+B*F) - alpha*P)
% 
% 
% % syms mp mc l theta theta_dot F g
% % syms x x_dot theta theta_dot
% % 
% % theta_ddot = (g*sin(theta)+cos(theta)*((-F-mp*l*theta_dot^2*sin(theta))/(mc+mp)))/(l*(4/3-(mp*cos(theta)^2)/(mc+mp)));
% % 
% % x_ddot = (F+mp*l*(sin(theta)*theta_dot^2)-cos(theta)*theta_ddot)/(mc+mp);
% % 
% % % dd_theta_theta_dot = jacobian(dd_theta, theta);
% % % dd_x_theta = jacobian(dd_x, theta);
% % 
% % jacobian(x_ddot, F)
% 
% 
% pP = zeros(2, 2);
% vP = zeros(2, 2);
% 
% % For position
% pP(1, 1) = P(1, 1);
% pP(2, 2) = P(3, 3);
% pP(1, 2) = P(1, 3);
% pP(2, 1) = P(1, 3);
% 
% % For velocity
% vP(1, 1) = P(2, 2);
% vP(2, 2) = P(4, 4);
% vP(1, 2) = P(2, 4);
% vP(2, 1) = P(2, 4);
% 
% [eig_vector, ~] = eig(pP);
% eig_value = eig(pP);
% 
% % Define theta vector
% theta = linspace(-pi, pi, 1000);
% ty1 = cos(theta) / sqrt(eig_value(1));
% ty2 = sin(theta) / sqrt(eig_value(2));
% ty = [ty1; ty2];
% tQ = inv(eig_vector');
% tx = tQ * ty;
% tx1 = tx(1, :);
% tx2 = tx(2, :);
% 
% % Plot safety envelope
% figure;
% plot(tx1, tx2, 'k', 'LineWidth', 2);
% hold on;
% line([x_set(1), x_set(2)], [theta_set(1), theta_set(1)],'color','k', 'LineWidth', 2);
% line([x_set(1), x_set(2)], [theta_set(2), theta_set(2)],'color','k', 'LineWidth', 2);
% line([x_set(1), x_set(1)], [theta_set(1), theta_set(2)],'color','k', 'LineWidth', 2);
% line([x_set(2), x_set(2)], [theta_set(1), theta_set(2)],'color','k', 'LineWidth', 2);
% 
% xlabel('x');
% ylabel('theta');
% title('Safety Envelope');
% grid on;

%% Normal Force
% x(1,:) = [ 0.0521393   4.91018112 -0.13274536 -5.93970777];
% x(2,:) = [ 0.10124111  4.90808848 -0.19214243 -5.96524678];
% x(3,:) = [ 0.15032199  4.87400485 -0.2517949  -5.93069499];
% x(4,:) = [ 0.19906204  4.81006071 -0.31110185 -5.84277671];
% x(5,:) = [ 0.24716265  4.71839965 -0.36952962 -5.70856779];
% x(6,:) = [ 0.29434664  4.60104972 -0.4266153  -5.53505255];
% x = x';
% 
% u = [-0.0874746259721774;
%     -3.2380344570381396;
%     -6.209824494432468;
%     -9.014868091242526;
%     -11.67280796931201;
%     -14.208055072731007];

%% Reduced Force
clear x
x(1,:) = [ 0.34812747  4.99835172 -0.17577963 -5.3536321];
x(2,:) = [ 0.39811098  5.03994946 -0.22931595 -5.48979124];
x(3,:) = [ 0.44851048  5.06753598 -0.28421386 -5.60496518];
x(4,:) = [ 0.49918584  5.08087891 -0.34026351 -5.69938815];
x(5,:) = [ 0.54999463  5.07982673 -0.3972574  -5.77371837];
x(6,:) = [ 0.60079289  5.06429381 -0.45499458 -5.82901209];
x = x';

u = [4.249143110292408;
    2.9063418504839404;
    1.542394967286011;
    0.1590685651634054;
    -1.2433792873849647;
    -2.6662722733675004];

N = 5;

y(1,:) = x(1,:);
y(2,:) = x(2,:);
y(3,:) = x(3,:);
y(4,:) = x(4,:);
y(5,:) = u';

Y = y(:,1:N);

X1 = x(1,2:6);
X2 = x(2,2:6);
X3 = x(3,2:6);
X4 = x(4,2:6);

% v1 = X1*Y'*pinv(Y*Y');
% P1 = v1*Y;
% 
% v2 = X2*Y'*pinv(Y*Y');
% P2 = v2*Y;
% 
% v3 = X3*Y'*pinv(Y*Y');
% P3 = v3*Y;
% 
% v4 = X4*Y'*pinv(Y*Y');
% P4 = v4*Y;

N = 8
Y = rand(5,N);
X = rand(1,N);

v = X*Y'*pinv(Y*Y')

P = v*Y;

figure;
plot([1:N], X(:), 'k', 'LineWidth', 1);
hold on;
plot([1:N], P(:), 'r', 'LineWidth', 1);

