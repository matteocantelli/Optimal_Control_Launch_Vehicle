% --- BLOCK 0: INITIAL SETUP ---

clear; 
clc; 
close all; 

%rng(6); % Set the random seed for reproducibility

% --- BLOCK 1: FILENAME ---

% Define the .mat filename (Fuel-Optimal) agent
agent_filename = 'Agent_rocket2.mat';

% --- BLOCK 2: CHECK FOR EXISTING FILE ---

% Check if the 'agent_filename' already exists.
if exist(agent_filename, 'file')
    % --- BLOCK 2A: LOAD AGENT (If it exists) ---
    fprintf('File found. Loading agent from %s...\n', agent_filename);
    load(agent_filename, 'agent', 'trainingStats'); % Load the saved agent

    % Redefine environment for simulation
    n = 7; % 7 states 
    obsInfo = rlNumericSpec([n 1], Name="State"); % Define the state info object

    % Redefine the action space EXACTLY as it was during training
    T_levels   = [0, 1200];                  % BANG-COAST (Off or Max)
    th_levels  = deg2rad(-10:5:10);          % 10 angles
    my_cell = {}; % Empty cell for actions
    for ii=1:numel(T_levels) % Thrust loop
        for jj=1:numel(th_levels) % Angle loop
            my_cell{end+1} = [T_levels(ii), th_levels(jj)]; % Add [T, th] pair
        end
    end
    actInfo = rlFiniteSetSpec(my_cell, Name="ThrustTheta"); % Action info object 

    % Create the environment for simulation
    env = rlFunctionEnv(obsInfo, actInfo, "step_rocket", "reset_rocket");
    fprintf('Loading complete. Ready for simulation.\n');

else
    % --- BLOCK 2B: TRAINING (If file does not exist) ---
    fprintf('File not found. Running full training (Fuel-Optimal)...\n');

    % --- BLOCK 3: OBSERVATION SPACE (Network Input) ---

    n = 7; % 7 states (normalized)
    obsInfo = rlNumericSpec([n 1], Name="State"); % Define the network's input

    % --- BLOCK 4: ACTION SPACE (Network Output) ---

    T_levels   = [0, 1200]; % 2 levels
    th_levels  = deg2rad(-10:5:10); % 5 levels
    my_cell = {}; % Empty cell
    for ii=1:numel(T_levels)
        for jj=1:numel(th_levels)
            my_cell{end+1} = [T_levels(ii), th_levels(jj)]; % Add action
        end
    end
    actInfo = rlFiniteSetSpec(my_cell, Name="ThrustTheta"); % 10 total actions
    fprintf('Action space created: %d actions.\n', length(actInfo.Elements));

    % --- BLOCK 5: NEURAL NETWORK (DQN) ---

    net = [
        featureInputLayer(n) % Input: 7
        fullyConnectedLayer(256)
        reluLayer
        fullyConnectedLayer(256)
        reluLayer
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(length(actInfo.Elements)) % Output: 10
    ];
    net = dlnetwork(net);

    % --- BLOCK 6: CRITIC OPTIMIZER OPTIONS ---

    criticOptions = rlOptimizerOptions(LearnRate=1e-4, GradientThreshold=1);

    % --- BLOCK 7: ENVIRONMENT AND AGENT ---

    env    = rlFunctionEnv(obsInfo, actInfo, "step_rocket", "reset_rocket");
    critic = rlVectorQValueFunction(net, obsInfo, actInfo);
    
    agentOpts = rlDQNAgentOptions(... % Starts the creation of options for the DQN agent.
        UseDoubleDQN=true, ... Prevents the agent from overestimating Q-values.
        TargetUpdateMethod="periodic", TargetUpdateFrequency=8, ...
        ExperienceBufferLength=200000, ... Sets the "memory" size.
        DiscountFactor=0.98, ... gives more weight to immediate rewards than future ones.
        MiniBatchSize=64, ... At each step, the agent will study a batch of 64 rand experiences from its memory.
        CriticOptimizerOptions=criticOptions); % Applies the optimizer options 
                                               % (like LearnRate=1e-4) that we defined earlier
    
    maxEps       = 2000;
    maxSteps     = 100;
    totalSteps   = maxEps * maxSteps;
    explorationSteps = 0.8 * totalSteps;
    epsilonMin   = 0.01; % Set minimum exploration chance to 1%.
    epsilonDecay = (1.0 - epsilonMin) / explorationSteps; % Calculates the amount to subtract from epsilon at each step 
                                                          % (linear decay) to go from 1.0 to 0.01 over 'explorationSteps'.
    agentOpts.EpsilonGreedyExploration.EpsilonDecay = epsilonDecay; % Apply the calculated decay rate.
    agentOpts.EpsilonGreedyExploration.EpsilonMin   = epsilonMin;   % Apply the calculated 1% minimum.
    
    agent = rlDQNAgent(critic, agentOpts);

    % --- BLOCK 8: TRAINING OPTIONS ---

    trainOpts = rlTrainingOptions(...
        MaxEpisodes=maxEps, ...
        MaxStepsPerEpisode=maxSteps, ...
        Plots="training-progress", ...
        StopTrainingCriteria="EpisodeCount", ...
        StopTrainingValue=maxEps, ... 
        UseParallel=false);

    % --- BLOCK 9: TRAIN ---

    trainingStats = train(agent, env, trainOpts); % Start the training process using the defined 
                                                  % agent, environment, and options.

    % --- BLOCK 10: SAVE ---

    fprintf('Training complete. Saving agent...\n');
    save(agent_filename, 'agent', 'trainingStats'); % Save the .mat file
end

% --- BLOCK 11: POST-TRAINING SIMULATION ---

fprintf('Running final simulation ...\n');
simOpts = rlSimulationOptions('NumSimulations', 1, 'StopOnError', 'off');
experience = sim(env, agent, simOpts);

% --- BLOCK 12: DATA EXTRACTION FOR PLOT ---

raw_obs = experience(1).Observation.State.Data;
obs = squeeze(raw_obs);
n = 7;
if size(obs,1)==n, obs_nT=obs; 
   elseif size(obs,2)==n, obs_nT=obs.'; 
   else, obs_nT=reshape(obs,[n,numel(obs)/n]); 
end

% De-normalize data for plotting
px = obs_nT(1,:) * 25.0;
py = obs_nT(2,:) * 100.0;
vx = obs_nT(3,:) * 50.0;
vy = obs_nT(4,:) * 50.0;
ey = obs_nT(6,:);
fuel_perc = obs_nT(7,:);
y_target = py - (ey*100.0);
x_target = 0*px;

u_raw = experience(1).Action.ThrustTheta.Data;
U = squeeze(u_raw);
if iscell(U), U = cell2mat(U.'); end
if size(U,1)~=2, U = U.'; end
T_applied = U(1,:);
theta_applied = U(2,:);

% --- BLOCK 13: PLOT RESULTS ---

figure;
subplot(2,2,1);
plot(px, py, 'b', 'LineWidth', 2); hold on;
yline(100, 'r--', 'LineWidth', 2);
title('Rocket Trajectory');
xlabel('Position x [m]'); ylabel('Altitude y [m]');
legend('Rocket', 'Target Altitude (100m)');
grid on; axis equal; xlim([-150 150]); % Set x-limits to see full range

subplot(2,2,2);
stairs(T_applied,'b','LineWidth',2);
hold on;
yyaxis right;
stairs(rad2deg(theta_applied),'r--','LineWidth',1.5);
yyaxis left;
ylabel('Thrust [N]'); ylim([-100, 2100]);
title('Action Policy'); grid on;

subplot(2,2,3);
plot(vx, 'LineWidth', 2); hold on;
plot(vy, 'LineWidth', 2);
title('Velocity'); ylabel('v [m/s]'); xlabel('time step');
legend('v_x (lateral)', 'v_y (vertical)'); grid on;

subplot(2,2,4);
plot(fuel_perc*100,'m','LineWidth',2);
title('Fuel Remaining'); ylabel('[%]'); xlabel('time step');
grid on; ylim([-10 110]);