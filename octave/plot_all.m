clear;
clc;
close all;

cd(fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'corrected'));

fprintf('Loading GreekFire results...\n');

%% =========================
% LOAD CSV DATA
% ==========================

equity = csvread('equity_curve.csv', 1, 0);
drawdown = csvread('drawdown.csv', 1, 0);
winrate = csvread('rolling_winrate.csv', 1, 0);
importance = csvread('feature_importance.csv', 1, 0);

%% =========================
% LOAD REGIME SIGNALS
% ==========================

regimes = {};

fid = fopen('regime_signal.csv', 'r');

if fid ~= -1

    fgetl(fid); % skip header

    line = fgetl(fid);

    while ischar(line)

        parts = strsplit(line, ',');

        if length(parts) >= 2
            regimes{end+1} = strtrim(parts{2});
        end

        line = fgetl(fid);
    end

    fclose(fid);

else
    warning('regime_signal.csv not found');
end

%% =========================
% LOAD MONTHLY ALPHA
% ==========================

alpha_step = [];
alpha_col = [];

fid = fopen('monthly_alpha.csv', 'r');

if fid ~= -1

    fgetl(fid); % skip header

    step_idx = 0;
    line = fgetl(fid);

    while ischar(line)

        step_idx = step_idx + 1;

        parts = strsplit(line, ',');

        % alpha_chosen column
        if length(parts) >= 5

            val = str2double(parts{5});

            if ~isnan(val)
                alpha_step(end+1) = step_idx;
                alpha_col(end+1) = val;
            end
        end

        line = fgetl(fid);
    end

    fclose(fid);

else
    warning('monthly_alpha.csv not found');
end

%% =========================
% FEATURE IMPORTANCE
% ==========================

fid = fopen('feature_importance.csv', 'r');

if fid == -1
    error('feature_importance.csv not found');
end

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

% Octave-safe nanmean
avg_imp = zeros(1, num_features);

for k = 1:num_features

    col = imp_matrix(:, k);
    col = col(~isnan(col));

    if isempty(col)
        avg_imp(k) = 0;
    else
        avg_imp(k) = mean(col);
    end
end

[sorted_avg, idx] = sort(avg_imp, 'descend');

top_n = min(10, num_features);

top_avg = sorted_avg(1:top_n);
top_names = headers(idx(1:top_n));

%% =========================
% MAIN PERFORMANCE FIGURE
% ==========================

figure( ...
    'Name', 'GreekFire Performance Analysis', ...
    'Position', [100 100 1400 900]);

%% -------------------------
% 1. EQUITY CURVE
% -------------------------

subplot(3,2,1);
hold on;

bull_col = [0.0 0.4 0.8];
bear_col = [1.0 0.5 0.0];
fallback_col = [0.0 0.7 0.3];
cash_col = [0.5 0.5 0.5];

% SPY benchmark
semilogy( ...
    equity(:,1), ...
    equity(:,3), ...
    'r', ...
    'LineWidth', 1.5);

n_eq = size(equity,1);
n_reg = length(regimes);
n_plot = min(n_eq, n_reg + 1);

% Fast grouped plotting
start_idx = 1;

for s = 2:n_plot

    changed = false;

    if s > n_reg
        changed = true;
    else
        changed = ~strcmp(regimes{s}, regimes{s-1});
    end

    if changed || s == n_plot

        reg = regimes{max(1, s-1)};

        if strcmp(reg, 'bull')
            col = bull_col;

        elseif strcmp(reg, 'bear')
            col = bear_col;

        elseif strcmp(reg, 'fallback')
            col = fallback_col;

        elseif strcmp(reg, 'cash')
            col = cash_col;

        else
            col = bull_col;
        end

        idx_range = start_idx:s;

        semilogy( ...
            equity(idx_range,1), ...
            equity(idx_range,2), ...
            'Color', col, ...
            'LineWidth', 2);

        start_idx = s;
    end
end

h1 = plot(NaN, NaN, ...
    'Color', bull_col, ...
    'LineWidth', 2);

h2 = plot(NaN, NaN, ...
    'Color', bear_col, ...
    'LineWidth', 2);

h3 = plot(NaN, NaN, ...
    'Color', fallback_col, ...
    'LineWidth', 2);

h4 = plot(NaN, NaN, ...
    'Color', cash_col, ...
    'LineWidth', 2);

h5 = plot(NaN, NaN, ...
    'r', ...
    'LineWidth', 1.5);

legend( ...
    [h1 h2 h3 h4 h5], ...
    {'Bull','Bear (Short)','Fallback','Cash','SPY'}, ...
    'Location','northwest');

title('Cumulative Returns (Regime Colored)');
xlabel('Step');
ylabel('Multiple');
grid on;

%% -------------------------
% 2. MONTHLY ALPHA
% -------------------------

subplot(3,2,2);

bar(alpha_step, alpha_col);

title('Actual 5-Day Excess Return (Alpha)');
xlabel('Step');
ylabel('Alpha');

grid on;

%% -------------------------
% 3. DRAWDOWN
% -------------------------

subplot(3,2,3);

plot( ...
    drawdown(:,1), ...
    drawdown(:,2), ...
    'r', ...
    'LineWidth', 2);

title('Strategy Drawdown');
xlabel('Step');
ylabel('Drawdown');

grid on;

%% -------------------------
% 4. ROLLING WIN RATE
% -------------------------

subplot(3,2,4);

plot( ...
    winrate(:,1), ...
    winrate(:,2), ...
    'g', ...
    'LineWidth', 2);

title('Rolling 12-Month Win Rate');
xlabel('Step');
ylabel('Win Rate');

ylim([0 1]);

grid on;

%% -------------------------
% 5. FEATURE IMPORTANCE
% -------------------------

subplot(3,2,[5 6]);

x = 1:top_n;

bar(x, top_avg);

title('Top Feature Importances (Average)');
xlabel('Feature');
ylabel('Importance Score');

set(gca, 'XTick', x);
set(gca, 'XTickLabel', top_names);

xtickangle(45);

grid on;

for i = 1:top_n

    text( ...
        x(i), ...
        top_avg(i) + 0.001, ...
        sprintf('%.3f', top_avg(i)), ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 8);
end

%% =========================
% FEATURE IMPORTANCE HEATMAP
% ==========================

figure( ...
    'Name', 'Feature Importance Over Time', ...
    'Position', [100 100 1400 600]);

top10_idx = idx(1:min(10, num_features));

imp_top10 = imp_matrix(:, top10_idx);
top10_names = headers(top10_idx);

imagesc(imp_top10);

colorbar;

title('Feature Importance by Step (Top 10 Features)');
xlabel('Feature');
ylabel('Step');

set(gca, ...
    'XTick', 1:min(10, num_features));

set(gca, ...
    'XTickLabel', top10_names);

xtickangle(45);

colormap('jet');

fprintf('\n=== Feature Importance Summary ===\n');
fprintf('Steps: %d | Features: %d\n', ...
    num_steps, num_features);

fprintf('\nTop 10 by Average Importance:\n');

for i = 1:top_n

    fprintf( ...
        '  %d. %s: %.4f\n', ...
        i, ...
        top_names{i}, ...
        top_avg(i));
end

fprintf('\nDone.\n');

%% =========================
% REGIME DIAGNOSTICS
% ==========================
diagfid = fopen('regime_diagnostics.csv', 'r');

if diagfid ~= -1

    diag_header = fgetl(diagfid);
    diag_cols = strsplit(diag_header, ',');

    diag_n_cols = length(diag_cols);

    diag_dates = {};
    diag_p_bull = [];
    diag_p_bear = [];
    diag_regimes = {};
    diag_raw_regimes = {};
    diag_vol = [];

    line = fgetl(diagfid);

    while ischar(line)

        parts = strsplit(line, ',');

        if length(parts) >= diag_n_cols

            diag_dates{end+1} = ...
                strtrim(parts{1});

            diag_p_bull(end+1) = ...
                str2double(parts{2});

            diag_p_bear(end+1) = ...
                str2double(parts{3});

            diag_regimes{end+1} = ...
                strtrim(parts{4});

            diag_raw_regimes{end+1} = ...
                strtrim(parts{5});

            diag_vol(end+1) = ...
                str2double(parts{6});
        end

        line = fgetl(diagfid);
    end

    fclose(diagfid);

    dn = length(diag_p_bull);
    dsteps = 1:dn;

    fprintf('\n=== Regime Diagnostics ===\n');
    fprintf('Total steps: %d\n', dn);

    bull_count = ...
        sum(strcmp(diag_regimes, 'bull'));

    bear_count = ...
        sum(strcmp(diag_regimes, 'bear'));

    cash_count = ...
        sum(strcmp(diag_regimes, 'cash'));

    fprintf( ...
        'Bull: %d (%.1f%%)\n', ...
        bull_count, ...
        100 * bull_count / dn);

    fprintf( ...
        'Bear: %d (%.1f%%)\n', ...
        bear_count, ...
        100 * bear_count / dn);

    fprintf( ...
        'Cash: %d (%.1f%%)\n', ...
        cash_count, ...
        100 * cash_count / dn);

    fprintf( ...
        'P(bull) range: [%.3f, %.3f], mean=%.3f\n', ...
        min(diag_p_bull), ...
        max(diag_p_bull), ...
        mean(diag_p_bull));

    fprintf( ...
        'P(bear) range: [%.3f, %.3f], mean=%.3f\n', ...
        min(diag_p_bear), ...
        max(diag_p_bear), ...
        mean(diag_p_bear));

    % Correct per-step max probability
    max_probs = max( ...
        [diag_p_bull(:), diag_p_bear(:)], ...
        [], 2);

    cash_steps = ...
        sum(max_probs < 0.55);

    fprintf( ...
        'max(P) < 0.55 (cash): %d steps (%.1f%%)\n', ...
        cash_steps, ...
        100 * cash_steps / dn);

    bear_steps = ...
        sum(diag_p_bear >= 0.60);

    fprintf( ...
        'P(bear) >= 0.60: %d steps (%.1f%%)\n', ...
        bear_steps, ...
        100 * bear_steps / dn);

    bull_steps = ...
        sum(diag_p_bull >= 0.60);

    fprintf( ...
        'P(bull) >= 0.60: %d steps (%.1f%%)\n', ...
        bull_steps, ...
        100 * bull_steps / dn);

    %% =========================
    % REGIME FIGURE
    % ==========================

    figure( ...
        'Name', 'Regime Diagnostics', ...
        'Position', [100 100 1400 900]);

    %% -------------------------
    % PANEL 1
    % HMM PROBABILITIES
    % -------------------------

    subplot(4,1,1);

    plot( ...
        dsteps, ...
        diag_p_bull, ...
        'b-', ...
        'LineWidth', 1.2);

    hold on;

    plot( ...
        dsteps, ...
        diag_p_bear, ...
        'r-', ...
        'LineWidth', 1.2);

    % Octave-safe threshold lines
    plot([1 dn], [0.65 0.65], '--k');
    plot([1 dn], [0.60 0.60], '--m');
    plot([1 dn], [0.55 0.55], '--g');
    plot([1 dn], [0.50 0.50], ':k');

    legend( ...
        'P(bull)', ...
        'P(bear)', ...
        'Location', ...
        'eastoutside');

    ylabel('Probability');
    title('HMM State Probabilities');

    grid on;
    xlim([1 dn]);

    %% -------------------------
    % PANEL 2
    % HMM INPUT VOLATILITY
    % -------------------------

    subplot(4,1,2);

    plot( ...
        dsteps, ...
        diag_vol, ...
        'k-', ...
        'LineWidth', 1.2);

    ylabel('Annualized Vol');

    title('SPY Volatility (HMM Input)');

    grid on;
    xlim([1 dn]);

    %% -------------------------
    % PANEL 3
    % REGIME STATE
    % -------------------------

    subplot(4,1,3);

    hold on;

    for s = 1:dn

        if strcmp(diag_regimes{s}, 'bull')

            c = [0.0 0.4 0.8];

        elseif strcmp(diag_regimes{s}, 'bear')

            c = [1.0 0.5 0.0];

        else

            c = [0.5 0.5 0.5];
        end

        rectangle( ...
            'Position', [s - 0.5, 0, 1, 1], ...
            'FaceColor', c, ...
            'EdgeColor', 'none');
    end

    ylim([0 1]);
    xlim([1 dn]);

    set(gca, 'YTick', []);

    title('Regime State');

    h1 = plot( ...
        NaN, NaN, ...
        's', ...
        'MarkerFaceColor', [0.0 0.4 0.8], ...
        'MarkerEdgeColor', 'none');

    h2 = plot( ...
        NaN, NaN, ...
        's', ...
        'MarkerFaceColor', [1.0 0.5 0.0], ...
        'MarkerEdgeColor', 'none');

    h3 = plot( ...
        NaN, NaN, ...
        's', ...
        'MarkerFaceColor', [0.5 0.5 0.5], ...
        'MarkerEdgeColor', 'none');

    legend( ...
        [h1 h2 h3], ...
        {'Bull','Bear','Cash'}, ...
        'Location', ...
        'eastoutside');

    %% -------------------------
    % PANEL 4
    % PROBABILITY DRIFT
    % -------------------------

    subplot(4,1,4);

    plot( ...
        dsteps, ...
        cumsum(diag_p_bull - 0.5), ...
        'b-', ...
        'LineWidth', 1.2);

    hold on;

    plot( ...
        dsteps, ...
        cumsum(diag_p_bear - 0.5), ...
        'r-', ...
        'LineWidth', 1.2);

    legend( ...
        'Cumulative P(bull)-0.5', ...
        'Cumulative P(bear)-0.5', ...
        'Location', ...
        'eastoutside');

    xlabel('Step');
    ylabel('Cumulative Drift');

    title('Probability Drift from 50/50');

    grid on;
    xlim([1 dn]);

    sgtitle( ...
        'GreekFire Regime Diagnostics', ...
        'FontSize', 14, ...
        'FontWeight', 'bold');

    print( ...
        '-dpng', ...
        '-r150', ...
        'regime_diagnostics.png');

    fprintf( ...
        'Saved regime_diagnostics.png\n');

else

    fprintf( ...
        '\nregime_diagnostics.csv not found — skipping diagnostics panel.\n');
end

fprintf('\nPlotting complete.\n');
