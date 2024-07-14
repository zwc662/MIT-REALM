function [K] = feedback_control(Ac, Bc, Ak, Bk, sc, sd)
%%%%%%%%%%%%%%%%%%%%%%  DOC HELP  %%%%%%%%%%%%%%%%%%%%%%
%% Inputs
%
%   Ac  : A(s) in continuous form  -- 4x4
%   Bc  : B(s) in continuous form  -- 4x1
%   Ak  : A(s) in discrete form    -- 4x4
%   Bk  : B(s) in discrete form    -- 4x1
%   sc  : current state vector     -- 4x1
%   sd  : desired state vector     -- 4x1
%
%% Outputs
%   K   : Feedback control gain    -- 1x4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%% Calc K %%%%%%%%%%%%%%%%%%%%%%%
    
    % s_star =  [0.14998542  4.84414696 -0.25146283 -5.87739881]';
    % s_star = [0.23434349, 0, -0.22644896, 0]';
    % u_star = -pinv(Bs)*As*s_star
    % u_star = -Bk'*Ak*s_star/(Bk'*Bk);
    
    % e_val = As*s_star + Bs*u_star
    % v = -pinv(Bk)*Ak*s_star
    n = 4;
    alpha = 0.96;
    
    val = abs(sc - sd)
    e = sc - sd

    D = [1/0.4,     0        0,       0;
             0,    1/4.5,       0,      0;
             0,      0,   1/0.4,      0;
             0,      0,       0,   1/4.5];

    % D = [1/0.4,     0        0,       0;
    %          0,    1/val(2),       0,      0;
    %          0,      0,   1/val(3),      0;
    %          0,      0,       0,   1/val(4)];

    % disp(Ac)
    % disp(Bc)
    % disp(Ak)
    % disp(Bk)
    cvx_begin sdp
    
        variable Q(n,n) symmetric;
        variable R(1,4);
    
        % minimize -log_det(Q)
    %     D * Q * D' - eye(4) < 0
        [alpha*Q,    Q*Ak' + R'*Bk';
          Ak*Q + Bk*R,    Q] >= 0
    %     [Q2,      R2';
    %      R2,      1/beta] > 0
        % [1   sd';
        %  sd   Q] >= 0

        % [1   e';
        %  e   Q] >= 0
    
        D * Q * D' - eye(4) <= 0
    
    cvx_end
    
    P = pinv(Q);
    K = R*P
    M = Ac + Bc*K;
    eig(M)
    assert(all(eig(M)<0))


end