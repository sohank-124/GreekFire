clear; clc; close all;
cd(fileparts(mfilename('fullpath')));

% Load numeric CSVs
equity = csvread('equity_curve.csv', 1, 0);
drawdown = csvread('drawdown.csv', 1, 0);
winrate = csvread('rolling_winrate.csv', 1, 0);
importance = csvread('feature_importance.csv', 1, 0);

% Load regime signal
fid = fopen('regime_signal.csv');
fgetl(fid); % skip header
regimes = {};
line = fgetl(fid);
while ischar(line)
    parts = strsplit(line, ',');
    if length(parts) >= 2
        regimes{end+1} = strtrim(parts{2});
    end
    line = fgetl(fid);
end
fclose(fid);

% Load monthly_alpha manually
fid = fopen('monthly_alpha.csv');
fgetl(fid);
alpha_col = [];
alpha_step = [];
step_idx = 0;
line = fgetl(fid);
while ischar(line)
    step_idx = step_idx + 1;
    parts = strsplit(line, ',');
    val = str2double(parts{end});
    if ~isnan(val)
        alpha_col(end+1) = val;
        alpha_step(end+1) = step_idx;
    end
    line = fgetl(fid);
end
fclose(fid);

% Read feature names from header
fid = fopen('feature_importance.csv');
header_line = fgetl(fid);
fclose(fid);
headers = strsplit(header_line, ',');
headers = headers(2:end);
keep = ~strcmp(headers, 'step_index');
headers = headers(keep);
imp_matrix = importance(:, 2:end);
imp_matrix = imp_matrix(:, keep);
num_steps = size(imp_matrix, 1);
num_features = size(imp_matrix, 2);

avg_imp = zeros(1, num_features);
for k = 1:num_features
    col_vals = imp_matrix(:, k);
    col_vals = col_vals(~isnan(col_vals));
    if ~isempty(col_vals)
        avg_imp(k) = mean(col_vals);
    end
end

[sorted_avg, idx] = sort(avg_imp, 'descend');
sorted_names = headers(idx);
top_n = min(10, num_features);
top_avg = double(sorted_avg(1:top_n));
top_names = sorted_names(1:top_n);

% Main figure
figure('Name', 'GreekFire Performance Analysis', 'Position', [100, 100, 1400, 900]);

% 1. Equity Curve with regime coloring
subplot(3, 2, 1);
hold on;

% Color map for regimes
% bull  = blue
% bear (shorting) = orange
% fallback = green  
% cash = grey

n_eq = size(equity, 1);
n_reg = length(regimes);
n_plot = min(n_eq, n_reg + 1); % equity has one extra point (starts at 1.0)

% Plot OEF (S&P 100) benchmark first
semilogy(equity(:,1), equity(:,3), 'r', 'LineWidth', 1.5);

% Plot strategy with regime colors segment by segment
for s = 1:(n_plot - 1)
    x_seg = [equity(s, 1), equity(s+1, 1)];
    y_seg = [equity(s, 2), equity(s+1, 2)];

    if s <= n_reg
        reg = regimes{s};
    else
        reg = 'bull';
    end

    if strcmp(reg, 'bull')
        col = [0.0, 0.4, 0.8];     % blue
    elseif strcmp(reg, 'bear')
        col = [1.0, 0.5, 0.0];     % orange (shorting active)
    elseif strcmp(reg, 'fallback')
        col = [0.0, 0.7, 0.3];     % green
    elseif strcmp(reg, 'cash')
        col = [0.5, 0.5, 0.5];     % grey
    else
        col = [0.0, 0.4, 0.8];     % default blue
    end

    semilogy(x_seg, y_seg, 'Color', col, 'LineWidth', 2);
end

% Legend patches
h1 = plot(NaN, NaN, 'Color', [0.0, 0.4, 0.8], 'LineWidth', 2);
h2 = plot(NaN, NaN, 'Color', [1.0, 0.5, 0.0], 'LineWidth', 2);
h3 = plot(NaN, NaN, 'Color', [0.0, 0.7, 0.3], 'LineWidth', 2);
h4 = plot(NaN, NaN, 'Color', [0.5, 0.5, 0.5], 'LineWidth', 2);
h5 = plot(NaN, NaN, 'r', 'LineWidth', 1.5);
legend([h1 h2 h3 h4 h5], {'Bull', 'Bear (Short)', 'Fallback', 'Cash', 'SPY'}, 'Location', 'northwest');

title('Cumulative Returns (Regime Colored)');
xlabel('Step'); ylabel('Multiple');
grid on;

% 2. Monthly Alpha
subplot(3, 2, 2);
bar(alpha_step, alpha_col);
title('Actual 5-Day Excess Return (Alpha)');
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

fprintf('\n=== Feature Importance Summary ===\n');
fprintf('Steps: %d | Features: %d\n', num_steps, num_features);
fprintf('\nTop 10 by Average Importance:\n');
for i = 1:top_n
    fprintf('  %d. %s: %.4f\n', i, top_names{i}, top_avg(i));
end
fprintf('\nDone.\n');