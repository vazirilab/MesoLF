function [C_out_raw, bg_temporal_out] = update_temporal_lasso( ...
    Y, A, C, bg_spatial, bg_temporal, IND, maxIter, sn, q)
%% update the temporal components using constrained HALS and oasis

% input:
%       Y:    d x T,  fluorescence data
%       A:    K element cell,  spatial components
%       C:    K element cell,  temporal components
% bg_spatial: d x 1,  spatial background
% bg_temporal: 1 x T, temporal backgrond
%     IND:    K element cell,  spatial extent for each component
%      sn:    d x 1,  noise std for each pixel
%       q:    scalar, control probability for FDR (default: 0.75), FDR for false discover rate?
%       lambda: calcium fitting regularizer
%       g:    time constant for auto-regressive model of Ca dynamics
% maxIter:    maximum HALS iteration (default: 40)
% options:    options structure

% output:
%   C_out_raw: d* T, updated temporal components with OASIS denoising
%   background term

% last update: 5/6/2020. YZ

%% options for HALS
%norm_C_flag = false;
tol = 1e-3;

% penalty related
if nargin < 9 || isempty(q)
    q = 0.75;
end

if nargin < 8 || isempty(sn)
    % nosie option
    sn_options.noise_range = [0.25, 0.7];
    sn_options.noise_method = 'logmexp';
    sn_options.block_size = [64, 64];
    sn_options.split_data = false;
    sn_options.max_timesteps = 3000;

    sn = get_noise_fft_mod(Y, sn_options);
end % sn is the average of power spectrum of Y along time domain

if nargin < 7 || isempty(maxIter)
    maxIter = 40;
end

%% cell to matrix
[size_h, size_w] = size(A{1});
if iscell(A)
    A_mat = zeros(size(Y, 1), length(A));
    for i = 1:length(A)
        A_mat(:, i) = A{i}(:);
    end
end

if iscell(C)
    C_mat = zeros(length(C), size(Y, 2));
    for i = 1:length(C)
        buf = C{i};
        C_mat(i, :) = buf(:).';
    end
end
[d, nr] = size(A_mat);
K = nr + size(bg_spatial, 2);
% here K is the neuron number + background component number, nr is the neuron number

%% combine
A_mat = [A_mat, bg_spatial];
C_mat = [C_mat; bg_temporal];

T = size(C_mat, 2);
sn = double(sn);

for i = 1:nr
    buf = A_mat(:, i);
    tmp_ind = IND{i};
    buf(~tmp_ind) = 0;
    A_mat(:, i) = buf;
end

%% HALS
[C_mat, ~] = HALS_temporal(Y, A_mat, C_mat, maxIter, tol, true, false);
C_out_raw = [];
for i = 1:length(C)
    buf = C_mat(i, :);
    C_out_raw{i} = buf(:);
end
f = C_mat(nr + 1:end, :); % separate f
bg_temporal_out = f;
