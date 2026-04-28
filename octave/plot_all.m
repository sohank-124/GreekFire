clear; clc; close all;

% Load numeric CSVs
equity = csvread('equity_curve.csv', 1, 0);
drawdown = csvread('drawdown.csv', 1, 0);
winrate = csvread('rolling_winrate.csv', 1, 0);
importance = csvread('feature_importance.csv', 1, 0);

% Load monthly_alpha manually (has string ticker column)
fid = fopen('monthly_alpha.csv');
fgetl(fid);
alpha_col = [];
line = fgetl(fid);
while ischar(line)
    parts = strsplit(line, ',');
    alpha_col(end+1) = str2double(parts{4});
    line = fgetl(fid);
end
fclose(fid);

% Read feature names from header
fid = fopen('feature_importance.csv');
header_line = fgetl(fid);
fclose(fid);
headers = strsplit(header_line, ',');
headers = headers(2:end); % skip first column (step_index)

% Remove any step_index column that snuck into features
keep = ~strcmp(headers, 'step_index');
headers = headers(keep);
imp_matrix = importance(:, 2:end);
imp_matrix = imp_matrix(:, keep);

num_steps = size(imp_matrix, 1);
num_features = size(imp_matrix, 2);

% Compute mean as plain numeric vector
avg_imp = zeros(1, num_features);
for k = 1:num_features
    col_vals = imp_matrix(:, k);
    col_vals = col_vals(~isnan(col_vals));
    if ~isempty(col_vals)
        avg_imp(k) = mean(col_vals);
    end
end

% Sort by mean importance
[sorted_avg, idx] = sort(avg_imp, 'descend');
sorted_names = headers(idx);

top_n = min(10, num_features);
top_avg = double(sorted_avg(1:top_n));
top_names = sorted_names(1:top_n);

% Main figure
figure('Name', 'GreekFire Performance Analysis', 'Position', [100, 100, 1400, 900]);

% 1. Equity Curve
subplot(3, 2, 1);
plot(equity(:,1), equity(:,2), 'b', 'LineWidth', 2); hold on;
plot(equity(:,1), equity(:,3), 'r', 'LineWidth', 2);
title('Cumulative Returns');
xlabel('Step'); ylabel('Multiple');
legend('Strategy', 'SPY Benchmark');
grid on;

% 2. Monthly Alpha
subplot(3, 2, 2);
bar(alpha_col);
title('Actual Monthly Excess Return');
xlabel('Step'); ylabel('Alpha');
grid on;

% 3. Drawdown
subplot(3, 2, 3);
plot(drawdown(:,1), drawdown(:,2), 'r', 'LineWidth', 2);
title('Strategy Drawdown');
xlabel('Step'); ylabel('Drawdown');
grid on;

% 4. Rolling Win Rate
subplot(3, 2, 4);
plot(winrate(:,1), winrate(:,2), 'g', 'LineWidth', 2);
title('Rolling 12-Month Win Rate');
xlabel('Step'); ylabel('Win Rate');
ylim([0 1]);
grid on;

% 5. Feature Importance
subplot(3, 2, [5, 6]);
x = 1:top_n;
bar(x, top_avg, 'FaceColor', [0.2, 0.5, 0.8]);
title('Top Feature Importances (Average)');
xlabel('Feature'); ylabel('Importance Score');
set(gca, 'XTick', x);
set(gca, 'XTickLabel', top_names);
xtickangle(45);
grid on;
for i = 1:top_n
    text(x(i), top_avg(i) + 0.001, ...
        sprintf('%.3f', top_avg(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8);
end

% Heatmap figure
figure('Name', 'Feature Importance Over Time', 'Position', [100, 100, 1400, 600]);
top10_idx = idx(1:min(10, num_features));
imp_top10 = imp_matrix(:, top10_idx);
top10_names = headers(top10_idx);
imagesc(imp_top10);
colorbar;
title('Feature Importance by Step (Top 10 Features)');
xlabel('Feature'); ylabel('Step');
set(gca, 'XTick', 1:min(10, num_features));
set(gca, 'XTickLabel', top10_names);
xtickangle(45);
colormap('jet');

% Summary
fprintf('\n=== Feature Importance Summary ===\n');
fprintf('Steps: %d | Features: %d\n', num_steps, num_features);
fprintf('\nTop 10 by Average Importance:\n');
for i = 1:top_n
    fprintf('  %d. %s: %.4f\n', i, top_names{i}, top_avg(i));
end
fprintf('\nDone.\n');
