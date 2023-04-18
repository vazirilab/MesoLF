function [adj_local_std, local_adjust, local_bg_s, local_bg_t, local_bg_block]...
    = local_bg_remove(N_num, img_stack, block_size)
%% this file would generate std image from the imaging dataset.
%  local rank_1
%  input: 
%       img_stack:  captured 2d image file
%       N_num: number of pixels under each microlens array
%       block_size: size of local neighbour (in units of microlens)
%  output:
%       adj_local_std: stiched local std after intensity adjustment
%       local_adjust: value of local adjust
%       local_bg_s: s from local rank-1 factorization
%       local_bg_t: t from local rank-1 factorization
%       local_bg_block: position(up left to down right) of each block.


%  last update: 2/13/2020. YZ

%% parameters
[size_h, size_w, frame_N] = size(img_stack);

%% block-by-block rank-1 detrend, non overlap
% calculate grid
N_grid_h = floor((size_h / N_num) / block_size);
if N_grid_h * block_size == size_h / N_num
    N_grid_h = N_grid_h - 1;
end
N_grid_w = floor((size_w / N_num) / block_size);
if N_grid_w * block_size == size_w / N_num
    N_grid_w = N_grid_w - 1;
end
% generate grid
x_ind = [(0 : 1 : N_grid_h) * block_size * N_num + 1, size_h];
y_ind = [(0 : 1 : N_grid_w) * block_size * N_num + 1, size_w];

[Y_ind, X_ind] = meshgrid(y_ind, x_ind);
% record local background
local_bg_s = cell(N_grid_h, N_grid_w);
local_bg_t = cell(N_grid_h, N_grid_w);
local_bg_block = cell(N_grid_h, N_grid_w);
local_std = cell(N_grid_h, N_grid_w);

bg_iter = 10;
for i = 1 : N_grid_h + 1
    i
    for j = 1 : N_grid_w + 1
        % grab the patch
        patch = img_stack(X_ind(i, j) : X_ind(i + 1, j + 1),...
            Y_ind(i, j) : Y_ind(i + 1, j + 1), :);
        patch_size = [size(patch, 1), size(patch, 2)];
        % conduct rank1 detrend
        temp_min = min(patch, [], [1, 2]);
        patch_sub_min = patch - temp_min;
        temp_max =  max(patch_sub_min(:));
        patch_sub_min = patch_sub_min / temp_max;

        [bg_spatial, bg_temporal] = rank_1_factorization(reshape(patch_sub_min, [], frame_N), bg_iter);
        
        % record      
        local_bg_s{i, j} = reshape(bg_spatial, patch_size);
        local_bg_t{i, j} = bg_temporal;
        local_bg_block{i, j} = [X_ind(i, j), Y_ind(i, j);...
           X_ind(i + 1, j + 1), Y_ind(i+1, j+1)]; % top left and bottom right corner
        
        std_img_rank1=compute_std_image(reshape(patch_sub_min, [], frame_N),bg_spatial, bg_temporal);
        std_img_rank1 = abs(std_img_rank1);
        std_img_rank1 = reshape(std_img_rank1, patch_size);
        local_std{i, j} = std_img_rank1;
        % record
    end
end

%% brightness adjustment, method 2
%  note the structure of the LFM image. calcualte the adjustment based on
%  the non-pixel mask

% local mask
local_adjust= cell(N_grid_h, N_grid_w);
x = ((1 : N_num) - ceil(N_num/2)) / N_num;
[Y, X] = meshgrid(x, x);
non_sigal_mask = (X.^2 + Y.^2 > 0.55^2);

adj_local_std = local_std;
% figure, imagesc(non_sigal_mask)
for i = 1 : N_grid_h + 1
    i
    for j = 1 : N_grid_w + 1
        % grab the std image
        current_img = local_std{i, j}; 
        if i == 1 && j == 1
            mask = zeros(size(current_img));
            mask(ceil(N_num/2) : N_num : end, ceil(N_num/2) : N_num : end) = 1;
            mask = conv2(mask, non_sigal_mask, 'same');
%             figure, imagesc(mask);axis equal, axis off
            record_ind = sum(mask .* current_img, 'all') / sum(mask, 'all');       
            local_adjust{1, 1} = 1;
            continue;
        end  
        

        % non signal mask
        mask = zeros(size(current_img));
        mask(ceil(N_num/2) : N_num : end, ceil(N_num/2) : N_num : end) = 1;
        mask = conv2(mask, non_sigal_mask, 'same');
%         
        avg_signal = sum(mask .* current_img, 'all') / sum(mask, 'all');
        local_adjust{i, j} = record_ind  / avg_signal;
       
       	% for debug
%         if i == 1 && j == N_grid_w + 1
%             figure, imagesc(mask); axis equal, axis off
%         end   
        adj_local_std{i, j} = adj_local_std{i, j} * local_adjust{i, j};
    end
end
