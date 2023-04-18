function [C_out, bg_temporal_out] = update_temporal_lasso_local(Y, A, C, bias,...
        bg_spatial, bg_temporal, IND, movie_size, maxIter, sn, q)
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
% maxIter:    maximum HALS iteration (default: 40)
% options:    options structure

% output: 
%   A_out: d*K, updated spatial components. Note taht K may contain the
%   background term

% update: add spike matrix as output

% last update: 4/1/2020. YZ

%% options for HALS
%norm_C_flag = false;
tol = 1e-3;
%repeat = 1;

if nargin < 9 || isempty(maxIter); maxIter = 40; end

if nargin<10 || isempty(sn)
    % nosie option
    sn_options.noise_range = [0.25,0.7];
    sn_options.noise_method = 'logmexp';
    sn_options.block_size = [64,64];
    sn_options.split_data = false;
    sn_options.max_timesteps = 3000;   
    sn = get_noise_fft_mod(Y,sn_options);  
end % sn is the average of power spectrum of Y along time domain

% penalty related
if nargin < 11 || isempty(q); q = 0.75; end

%% cell to matrix
size_h = movie_size(1);
size_w = movie_size(2);

if iscell(A)
    A_mat = zeros(size(Y, 1), length(A));
    for i = 1 : length(A)
        buf = zeros(size_h, size_w);
        buf_bias = bias{i};
        buf(buf_bias(1, 1) : buf_bias(2, 1), buf_bias(1, 2) : buf_bias(2, 2)) = gather(A{i});
        A_mat(:, i) = buf(:);
    end
end

if iscell(IND)
    IND_mat = false(size(Y, 1), length(IND));
    for i = 1 : length(A)
        buf = false(size_h, size_w);
        buf_bias = bias{i};
        buf(buf_bias(1, 1) : buf_bias(2, 1), buf_bias(1, 2) : buf_bias(2, 2)) = gather(IND{i});
        IND_mat(:, i) = buf(:);
    end
end

if iscell(C)
    C_mat = zeros(length(C), size(Y, 2));
    for i = 1 : length(C)
        buf = C{i};
        C_mat(i, :) = gather(buf(:).');
    end
end
[~, nr] = size(A_mat);
%K = nr + size(bg_spatial, 2); 
% here K is the neuron number + background component number, nr is the neuron number

%% combine
A_mat = [A_mat, bg_spatial];
C_mat = [C_mat; bg_temporal];

%T = size(C_mat,2);
%sn = double(sn);

for i = 1 : nr
    buf = A_mat(:, i);
    tmp_ind = IND_mat(:, i);
    buf(~tmp_ind) = 0;
    A_mat(:, i) = buf;
end

%% HALS
[C_mat, ~] = HALS_temporal(Y, A_mat, C_mat, maxIter, tol, true, false);

%% OASIS
% last update temporal
% 
% decimate = 1;
% % optimize_b = false;
% % optimize_g = true;
% % maxIter = 100; % which is only been used to get gama and background
% gama = 0.9;
% % lam = 0.05;
% maxIter_oasis = 100;
% Spike_mat = zeros(nr, size(C_mat, 2));
% for i = 1 : nr
% 
%     y = C_mat(i, :);
%     [c, s, b, g, active_set] = foopsi_oasisAR1(y, gama, lam, false,...
%         true, decimate, maxIter_oasis);
%     
%     % 
% %     figure(102), plot(c)
% %     figure(103), plot(y)
%     C_mat(i, :) = c;
%     Spike_mat(i, :) = s;
% end

%% update the variable
% neuron
C_out = cell(length(C), 1);
for i = 1 : length(C)
    buf = C_mat(i, :);
    C_out{i} =buf(:);
end
% background
f = C_mat(nr+1:end,:); % separate f
bg_temporal_out = f;
%disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Memory usage in update_temporal_lasso_local(): ']); whos;
end
