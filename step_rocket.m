function [NextObs, Reward, IsDone, LoggedSignals] = step_rocket(Action, LoggedSignals)

% Description: Executes one simulation step with Shaping Reward

    % --- BLOCK 1: UNPACK ---

    x_k = LoggedSignals.State;    
    kk  = LoggedSignals.curr_time; 
    Ts  = 0.1; 
    
    % Action decoding
    Action = Action'; 
    if iscell(Action), Action = Action{1}; 
    end 
    a = double(Action(:)');  
    T_real = a(1);           
    theta_real = a(2);       

    % --- BLOCK 2: PHYSICS PREP (WIND FIX) ---
   
    wind_x = LoggedSignals.disturbance_sigma * randn();
    wind_y = LoggedSignals.disturbance_sigma * randn();
    
    params.g_real   = LoggedSignals.g_real;
    params.c_d_real = LoggedSignals.c_d_real;
    params.Isp      = LoggedSignals.Isp;
    params.m_dry    = LoggedSignals.m_dry;
    params.wind_current = [wind_x; wind_y]; 
    
    % --- BLOCK 3: INTEGRATION ---
    
    [~, X] = ode45(@(t,x) my_physics_2D(t, x, [T_real, theta_real], params), [kk*Ts, (kk+1)*Ts], x_k);
    
    next_state = X(end, :)';           
    LoggedSignals.State = next_state;  

    px = next_state(1);  
    py = next_state(2);
    vx = next_state(3);  
    vy = next_state(4);
    m  = next_state(5);

    % --- BLOCK 4: OBSERVATION & ERRORS ---

    ex = px - LoggedSignals.x_target;
    ey = py - LoggedSignals.y_target;
    
    % Distanza Euclidea Attuale (per il Shaping)
    dist_curr = sqrt(ex^2 + ey^2);
    
    fuel_perc  = (m - LoggedSignals.m_min) / (LoggedSignals.m_initial - LoggedSignals.m_min);
    fuel_perc  = max(0, min(1, fuel_perc)); 
    
    % Normalize
    NextObs = [px/25; py/100; vx/50; vy/50; ex/25; ey/100; fuel_perc];

    % --- BLOCK 5: TERMINATION ---

    r_tol = 3.0;   
    v_tol = 3.0;   
    
    % Successo: Vicino al target E lento
    is_stable = (sqrt(vx^2 + vy^2) < v_tol);
    goal_reached = (abs(ey) <= r_tol) && (abs(ex) <= 5.0) && is_stable;
    
    failure_crash = (py < -1.0) ...             
                 || (py > 150) ...              
                 || (abs(px) > 25);            
                 
    failure_fuel  = (m <= LoggedSignals.m_min); 
    time_out      = (kk >= 100); 
    
    IsDone = goal_reached || failure_crash || failure_fuel || time_out;

    % --- BLOCK 6: REWARDS ---

    Reward = 0;

    % 6A. Distance Shaping 
    dist_prev = LoggedSignals.prev_dist;
    shaping_reward = (dist_prev - dist_curr) * 1.0; 
    Reward = Reward + shaping_reward;

    Reward = Reward - (0.01 * abs(vy));

    % 6B. Fuel Cost
    if T_real > 0
       Reward = Reward - LoggedSignals.R_u; 
    end

    % 6C. Engine Penality 
    prev_T = LoggedSignals.LastAction(1); 

    if (T_real > 0) && (prev_T == 0)

        Reward = Reward - 5.0; 

    end
    
    % 6D. Terminal Rewards
    if IsDone
        if goal_reached
            Reward = Reward + LoggedSignals.R_goal; 
            if (abs(ex) < 2.0) && (abs(ey) < 2.0)
                Reward = Reward + 50.0;
                fprintf('GOAL REACHED at step %d! Fuel: %.1f%%\n', kk, fuel_perc*100);
            end
        elseif failure_crash
            if py>150.0
                Reward=Reward - 20.0;
            else 
            Reward = Reward - LoggedSignals.R_fail; % -10    
            end
        elseif failure_fuel
            Reward=Reward - LoggedSignals.R_fail;
        end
    end

    % --- BLOCK 7: UPDATE MEMORY ---

    LoggedSignals.curr_time = kk + 1;
    LoggedSignals.LastAction = [T_real, theta_real];
    LoggedSignals.prev_dist = dist_curr; 
end