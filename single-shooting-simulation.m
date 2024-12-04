% "Design and implementation of a ball-balancing platform 
%  using 3-DOF RPS manipulator"
%
% Single Shooting Trajectory Calculation
%
% MAE 589 - Optimal Controls and Reinforcement Learning
%
% Authors: Karthik Kannan, Dhruv Joshi, Soham Gotkhindikar

clear;
clc;
close all;

import casadi.*
% temporal discretization
dt = 0.1;
T = 12;
N = T / dt + 1

% define dynamics
x1 = SX.sym('x1');
x2 = SX.sym('x2');
x3 = SX.sym('x3');
x4 = SX.sym('x4');
a1 = SX.sym('a1');
a2 = SX.sym('a2');
x = [x1; x2; x3; x4];
a = [a1; a2];
xdot = [
    x3*cos(x4);
    x3*sin(x4);
    a1
    a2
];
traj = Function('traj', {x, a}, {xdot});

% initial and final state
init_state = [300; 300; 0; 0];
end_state = [-300; 300; 0; 0];

% define state vectors
X = SX.sym('X', 4, N+1);
A = SX.sym('A', 2, N);

X(:,1) = init_state;

% define obstacles: [x, y, radius]
o1 = [-100, 200, 100];
o2 = [100, 200, 100];

cost = 0;
obst_cons = [];

% integrate cost and state functions
for t = 1:N
    % update dynamics
    X(:,t+1) = X(:,t) + dt*traj(X(:,t), A(:,t));
    % compute stage cost
    cost = cost + dt * ((X(1,t) - 3)^2 + (X(2,t) - 3)^2);
    % compute distances
    dist1 = sqrt( (X(1,t) - o1(1) )^2 + ( X(2,t) - o1(2) )^2 );
    dist2 = sqrt( (X(1,t) - o2(1) )^2 + ( X(2,t) - o2(2) )^2 );
    % add obstacle constraint
    obst_cons = [obst_cons; o1(3) - dist1; o2(3) - dist2];
end

% compute terminal cost
lambda = 30;
cost = cost + lambda * ((X(1,end) - end_state(1))^2 + (X(2,end) - end_state(2))^2) + lambda * ((X(3,end)^2 + X(4,end)^2)) + lambda;

% add combined constaint function
control_cons = [A(:)];

% set bounds
lbg_control = -100 * ones(2*N,1);
ubg_control = 100 * ones(2*N,1);
lbg_obstacle = -inf(size(obst_cons));
ubg_obstacle = zeros(size(obst_cons));
lbg = [lbg_obstacle; lbg_control];
ubg = [ubg_obstacle; ubg_control];

g = [obst_cons; control_cons];

% solver
nlp = struct('x', A(:), 'f', cost, 'g', g);
opti = struct; 
opti.ipopt.max_iter = 100000;

solver = nlpsol('solver', 'ipopt', nlp, opti);
A_init = zeros(2, N);
sol = solver('x0', A_init(:), 'lbg', lbg, 'ubg', ubg);
A_opt = reshape(full(sol.x), 2, N);
X_opt = zeros(4, N+1);
X_opt(:, 1) = init_state;

for t = 1:N
    X_opt(:,t+1) = X_opt(:,t) + full(dt*traj(X_opt(:,t), A_opt(:,t)));
end

time = linspace(0, T, N+1);

%% Plotting
figure;
subplot(2,1,1);
plot(time, X_opt(1,:), 'r','LineWidth', 2); hold on;
plot(time, X_opt(2,:), 'b', 'LineWidth',2);
xlabel('Time [s]');
legend('x1 (horizontal)', 'x2 (vertical)');
title('Optimal State Trajectory');
grid on;

subplot(2,1,2);
plot(time, X_opt(3,:), 'g','LineWidth', 2); hold on;
plot(time, X_opt(4,:), 'm', 'LineWidth',2);
xlabel('Time [s]');
legend('x3 (velocity)', 'x4 (angle)');
title('Velocity and Angle Trajectory');
grid on;

figure;
time_ctrl = linspace(0, T-dt, N);
subplot(2,1,1);
plot(time_ctrl, A_opt(1,:), 'r', 'LineWidth', 2);
xlabel('Time [s]');
ylabel('Control Input u1 (Acceleration)');
title('Optimal Control Input: u1');
grid on;

subplot(2,1,2);
plot(time_ctrl, A_opt(2,:), 'b', 'LineWidth', 2);
xlabel('Time [s]');
ylabel('Control Input u2 (Angular velocity)');
title('Optimal Control Input: u2');
grid on;

figure;
plot(X_opt(1,:),X_opt(2,:));
xlabel('x1 (horizontal)');
ylabel('x2 (vertical)');
title('Optimal State Trajectory in (x1, x2) space');
axis equal;
hold on;

grid on;
optimal_cost = full(sol.f);
disp(['Optimal Cost: ', num2str(optimal_cost)]);

theta = linspace(pi, 2*pi, 100); 
r = 125;                       
h = 0;                        
k = 200;                       
x = h + r * 2*cos(theta);      
y = k + r * 0.8*sin(theta);      
plot(x, y, 'r', 'LineWidth', 2);
