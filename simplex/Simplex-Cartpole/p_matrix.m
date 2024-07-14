


A = [1,    0.03333333,    0,            0;
     0,    1,             -0.05649123,  0;
     0,    0,             1,            0.03333333;
     0,    0,             0.89802632,   1        ];

B = [0; 0.03341688; 0; -0.0783208];
 
%D = [1/0.8,   0,        0,        0;
%       0,     0,       1/0.8,     0;
%       0,     1/100,    0,         0;
%       0,     0,       0,     1/100];

D = [1/0.7,   0,        0,        0;
        0,     0,       1/0.6,     0];


n = 4;
alpha = 0.865
beta = 0.002
invV = 1/beta;
c = 1/15

cvx_begin sdp

    variable Q(n,n) symmetric;
    variable R(1,4);

    minimize -log_det(Q)
    D * Q * D' - eye(2) < 0
    [alpha*Q,      Q*A' + R'*B';
      A*Q + B*R,  Q] > 0
    [Q,      R';
     R,      1/beta] > 0


cvx_end

P = inv(Q)
F = R*P
A + B*F

check = eig((A+B*F)'*P*(A+B*F) - alpha*P)





