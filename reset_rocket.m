function [InitialObservation, LoggedSignals] = reset_rocket()

% Description: Resets the environment with improved Reward Shaping setup

    % --- BLOCK 1: INITIAL PHYSICAL STATE ---

    px0 = -20 + (40 * rand);  % Random lateral position
    py0 = 0;                  % Start from ground
    vx0 = 0;                  % Random lateral velocity
    vy0 = 0;                  % Start at rest vertically
    m0  = 50.0;               % Initial mass
    
    LoggedSignals.State = [px0; py0; vx0; vy0; m0]; 
    LoggedSignals.curr_time = 1; 
    LoggedSignals.LastAction = [0,0]; 

    % --- BLOCK 2: PHYSICAL PARAMETERS ---

    LoggedSignals.g_real   = 9.81;          
    LoggedSignals.c_d_real = 0.1;           
    LoggedSignals.Isp      = 250;           
    LoggedSignals.disturbance_sigma = 10.0;  
    LoggedSignals.m_dry    = 35.0;          % 50 - 15
    LoggedSignals.m_min    = 40.0;          % 50 - 10 (Budget 10kg)
    LoggedSignals.m_initial= 50.0;          
    LoggedSignals.T_max    = 2000.0;           
     
    % --- BLOCK 3: REWARD PARAMETERS ---

    LoggedSignals.R_lazy = 0.1;      % Penality for sitting on the ground
    LoggedSignals.R_u    = 0.05;     % Penality for using fuel
    LoggedSignals.R_goal = 30.0;     % Reward for success
    LoggedSignals.R_fail = 10.0;     % Penality crash
    LoggedSignals.R_x    = 0.5;      % Penality for wrong position
    
    % --- BLOCK 4: TARGET DEFINITION ---

    LoggedSignals.x_target = 0.0;
    LoggedSignals.y_target = 100.0;
    
    % --- BLOCK 5: MEMORY FOR SHAPING ---

    dist_x = px0 - LoggedSignals.x_target;
    dist_y = py0 - LoggedSignals.y_target;
    current_dist = sqrt(dist_x^2 + dist_y^2);
    
    LoggedSignals.prev_dist = current_dist; 

    % --- BLOCK 6: INITIAL OBSERVATION ---

    ex0 = dist_x;
    ey0 = dist_y;
    fuel_perc0 = (m0 - LoggedSignals.m_min) / (LoggedSignals.m_initial - LoggedSignals.m_min);
    
    % Normalize
    px_norm = px0 / 25.0;
    py_norm = py0 / 100.0;
    vx_norm = vx0 / 50.0;
    vy_norm = vy0 / 50.0;
    ex_norm = ex0 / 25.0;
    ey_norm = ey0 / 100.0;
    
    InitialObservation = [px_norm; py_norm; vx_norm; vy_norm; ex_norm; ey_norm; fuel_perc0];
end