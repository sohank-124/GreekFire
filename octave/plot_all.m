% plot_all.m - GNU Octave script to visualize GreekFire results

% Clear environment
clear; clc; close all;

% Load data
equity = csvread('equity_curve.csv', 1, 0);
alpha = csvread('monthly_alpha.csv', 1, 0);
importance = csvread('feature_importance.csv', 1, 0);
drawdown = csvread('drawdown.csv', 1, 0);
winrate = csvread('rolling_winrate.csv', 1, 0);

figure('Name', 'GreekFire Performance Analysis', 'Position', [100, 100, 1200, 800]);

% 1. Equity Curve
subplot(3, 2, 1);
plot(equity(:, 1), equity(:, 2), 'b', 'LineWidth', 2); hold on;
plot(equity(:, 1), equity(:, 3), 'r', 'LineWidth', 2);
title('Cumulative Returns');
xlabel('Step'); ylabel('Multiple');
legend('Strategy', 'SPY Benchmark');
grid on;

% 2. Monthly Alpha
subplot(3, 2, 2);
bar(alpha(:, 4))
title('Actual Monthly Excess Return');
xlabel('Step'); ylabel('Alpha');
grid on;

% 3. Drawdown
subplot(3, 2, 3);
plot(drawdown(:, 1), drawdown(:, 2), 'r', 'LineWidth', 2);
title('Strategy Drawdown');
xlabel('Step'); ylabel('Drawdown');
grid on;

% 4. Rolling Win Rate
subplot(3, 2, 4);
plot(winrate(:, 1), winrate(:, 2), 'g', 'LineWidth', 2);
title('Rolling 12-Month Win Rate');
xlabel('Step'); ylabel('Win Rate');
ylim([0 1]);
grid on;

% 5. Feature Importance (Top 10 avg)
subplot(3, 2, [5, 6]);
% Calculate mean importance across all steps (columns 2 onwards in CSV)
imp_matrix = importance(:, 2:end);
avg_imp = mean(imp_matrix, 1);
[sorted_imp, idx] = sort(avg_imp, 'descend');
num_features = min(10, length(sorted_imp));
bar(sorted_imp(1:num_features));
title('Top Feature Importances (Average)');
xlabel('Feature Rank'); ylabel('Importance Score');
grid on;

fprintf('Visualization complete. Please check the figure window.\n');