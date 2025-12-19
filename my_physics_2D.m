function x_dot = my_physics_2D(t, x, u, params)

% Description: Simulates the 2D rocket dynamics (Equations of Motion).

    % --- BLOCK 1: PARAMETERS ---

    g   = params.g_real;   % Gravity
    c_d = params.c_d_real; % Drag Coefficient
    Isp = params.Isp;      % Specific Impulse
    g0  = 9.81;            % Standard gravity (for Isp)
    
    Force_disturbance = params.wind_current;

    % --- BLOCK 2: STATES ---
   
    px = x(1); 
    py = x(2); 
    vx = x(3);
    vy = x(4);
    m  = x(5);

    % --- BLOCK 3: ACTION ---

    % u = [T, theta]
    T     = u(1); % Thrust (Real)
    theta = u(2); % Angle (Real, rad)

    % --- BLOCK 4: DYNAMICS (Equations of Motion) ---

    % 4A. Thrust Components (theta from vertical)
    Tx = T * sin(theta);
    Ty = T * cos(theta);

    % 4B. Vector Drag
    v_vec = [vx; vy];
    v_norm = norm(v_vec);              % Magnitude |v|
    Drag_vec = c_d * v_vec * v_norm;   % F_drag = c_d * |v| * v

    % 4D. Position Derivatives
    dpxdt = vx;
    dpydt = vy;

    % 4E. Velocity Derivatives (a = F_net / m)
    dvxdt = (Tx - Drag_vec(1) + Force_disturbance(1)) / m;
    dvydt = (Ty - Drag_vec(2) - m*g + Force_disturbance(2)) / m;
    
    % 4F. Mass Derivative
    dmdt  = -abs(T) / (Isp * g0);

    % 4G. Safety constraint (do not go below m_dry)
    if (m + dmdt*0.01 < params.m_dry), dmdt = 0; 
    end

    % --- BLOCK 5: OUTPUT ---

    x_dot = [dpxdt; dpydt; dvxdt; dvydt; dmdt]; % Return the derivatives
end