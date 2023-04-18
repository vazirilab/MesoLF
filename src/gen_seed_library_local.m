function [S, T, S_mask, bias] = gen_seed_library_local(PSF_struct, valid_seg, ...
    sensor_movie, bg_spatial, bg_temporal, gpu_device)
%% generate the spatial iteration components used by MesoLF. Only forward 
%  propagation is required, thus the speed could be fast.
%  fast version, disable per segmentation check.

%  use text command to label the clusterred component
%  arrange different color
%  Input:
%       PSF_struct: structure that containes LFM psf and other info
%       valid_seg: spatial components segments
%       valid_seg: valid segmentation
%       bg_spatial: spatial background, low rank
%       bg_temporal: temporal background, low rank


%  Output:
%       S_init: cell array, contains grascale mask for each segemnts
%           and a binary mask 
%       T_init


%  update:
%       save the split component in target folder, with colorful label
%       for the component check: use skip recon. after check, use full
%       recon

%  last update: 4/23/2020. YZ

%% configuration

% LFM
recon_step = 5;
H = PSF_struct.H;
Ht = PSF_struct.Ht;
Nnum = size(H ,3);
[size_h, size_w] = size(bg_spatial);
size_z = size(H ,5);
if size(bg_spatial, 3) > 1 || size(bg_temporal, 1) > 1
    error('require rank-1 background')
end

N_seg = size(valid_seg, 1);

if isa(gpu_device, 'parallel.gpu.CUDADevice')
    xsize = [size_h, size_w];
    msize = [size(H,1), size(H,2)];
    mmid = floor(msize/2);
    exsize = xsize + mmid;  % to make the size as 2^N after padding
    exsize = [ min( 2^ceil(log2(exsize(1))), 128*ceil(exsize(1)/128) ), min( 2^ceil(log2(exsize(2))), 128*ceil(exsize(2)/128) ) ];    
    zeroImageEx = gpuArray(zeros(exsize, 'single'));
    
    backwardFUN = @(projection) backwardProjectGPU(Ht, projection, zeroImageEx, exsize);
    forwardFUN = @(Xguess) forwardProjectGPU( H, Xguess, zeroImageEx, exsize); % one use H and one use Ht
    forwardFUN_step = @(Xguess, z_depth) forwardProjectGPU_skip_2D( H, Xguess, zeroImageEx, exsize, recon_step, z_depth);
else
	forwardFUN =  @(Xguess) forwardProjectACC( H, Xguess, CAindex ); % build the function: forward and backward
    backwardFUN = @(projection) backwardProjectACC(Ht, projection, CAindex ); 
end

%% calculate PSF range
margin = max(20, Nnum);
for i = 1 : size(H, 5)
    buf = sum(H(:, :, :, :, i), [3, 4]);
    buf = buf(round(end / 2), round(end / 2) : end);
    idx = find(buf < max(buf) * 0.1);
    idx = idx(1);
   
    H_range(i) = min((idx + margin) * 2 + 1, size(H, 1));
end
% figure, plot(H_range)
%% main for
disp('build forward component')
% detrend
sensor_movie_de_bg = sensor_movie - reshape(bg_spatial, [], 1) * bg_temporal;
sensor_movie_de_bg = reshape(sensor_movie_de_bg, size_h, size_w, []);
% define spike distance

S_init = [];
T_init = [];
S_mask_init = [];
component_num = 1;
fprintf('start build component...\n')
h_fig = figure(102);
% segment-by-segment loop
for i = 1 : N_seg% loop 1
    
    fprintf('\t build component No.%d...  \n', i)
    % read all patches and positions from valid_seg
    curr_seg = valid_seg{i, 1}; % still a cell array
    curr_pos = valid_seg{i, 2}; % a matrix
    num_comp_in_seg = length(curr_seg);
    patch_size = size(curr_seg{1}, 1);
    
    % depth bin     
    depth_array = curr_pos(:, 3);
    [unique_array, ia, ic] = unique(depth_array);

    % maximum bias from NIP
    [~, max_size_PSF_ind] = max(abs(depth_array - size(H, 5) / 2));
    max_PSF_size = H_range(depth_array(max_size_PSF_ind));
    
    % segnents lateral range estimation
    top_left_corner(1) = min(curr_pos(:, 1));
    top_left_corner(2) = min(curr_pos(:, 2));
    bottom_right_corner(1) = max(curr_pos(:, 1));
    bottom_right_corner(2) = max(curr_pos(:, 2));
       
    top_left_corner = round(max(top_left_corner - max_PSF_size / 2, 1));
    % align with the Nnum
    top_left_corner = top_left_corner - mod(top_left_corner, Nnum) + 1;
    
    bottom_right_corner = round(bottom_right_corner + max_PSF_size / 2);
    
    size_buf = bottom_right_corner - top_left_corner + 1; 
    size_buf = ceil(size_buf / Nnum) * Nnum;
    
    % boundary limit
    bottom_right_corner(1) = min(top_left_corner(1) + size_buf(1) - 1, size_h);
    bottom_right_corner(2) = min(top_left_corner(2) + size_buf(2) - 1, size_w);
    size_buf = bottom_right_corner - top_left_corner + 1;
    
    % build small volume
    sample_3D = zeros(size_buf(1), size_buf(2), length(unique_array));
    sample_mask_3D = zeros(size_buf(1), size_buf(2), length(unique_array));
    
    % convolution parameter
	xsize_crop = [size_buf(1), size_buf(2)];
    msize_crop = [max_PSF_size, max_PSF_size];
    mmid_crop = floor(msize_crop/2);
    exsize_crop = xsize_crop + mmid_crop;  % to make the size as 2^N after padding
    exsize_crop = [ min( 2^ceil(log2(exsize_crop(1))), 128*ceil(exsize_crop(1)/128) ),...
        min( 2^ceil(log2(exsize_crop(2))), 128*ceil(exsize_crop(2)/128) ) ];    
    zeroImageEx_crop = gpuArray(zeros(exsize_crop, 'single'));  
    
    sample_3D_debug = zeros(size_h, size_w, size_z);
    % collect different segments
    for kk = 1 : num_comp_in_seg
        buf2 = zeros(size_h, size_w);
        buf2(curr_pos(kk, 1) - floor(patch_size / 2) : curr_pos(kk, 1) + floor(patch_size / 2), ...
          curr_pos(kk, 2)  - floor(patch_size / 2) : curr_pos(kk, 2)+ floor(patch_size / 2)) =  curr_seg{kk};
        sample_3D(:, :, ic(kk)) = sample_3D(:, :, ic(kk)) + ...
            buf2(top_left_corner(1) : bottom_right_corner(1), top_left_corner(2) : bottom_right_corner(2));

        buf2 = zeros(size_h, size_w);        
        buf2(curr_pos(kk, 1) - floor(patch_size / 2) : curr_pos(kk, 1) + floor(patch_size / 2), ...
          curr_pos(kk, 2)  - floor(patch_size / 2) : curr_pos(kk, 2)+ floor(patch_size / 2)) = 1; 
        sample_mask_3D(:, :, ic(kk)) = sample_mask_3D(:, :, ic(kk)) + ...
            buf2(top_left_corner(1) : bottom_right_corner(1), top_left_corner(2) : bottom_right_corner(2));
        
        % for debug, output oringinal volume
        %{
        sample_3D_debug(curr_pos(kk, 1) - floor(patch_size / 2) : curr_pos(kk, 1) + floor(patch_size / 2), ...
          curr_pos(kk, 2)  - floor(patch_size / 2) : curr_pos(kk, 2)+ floor(patch_size / 2), ...
          curr_pos(kk, 3) ) = ...
          sample_3D_debug(curr_pos(kk, 1) - floor(patch_size / 2) : curr_pos(kk, 1) + floor(patch_size / 2), ...
          curr_pos(kk, 2)  - floor(patch_size / 2) : curr_pos(kk, 2)+ floor(patch_size / 2), ...
          curr_pos(kk, 3) ) + curr_seg{kk};   
        %}
    end         
    % spatial initialization
%     spatial_init_debug = gather(forwardFUN(sample_3D_debug));
%     figure, imagesc(sum(sample_3D, 3))
%     figure, imagesc(sum(sample_3D_debug, 3))
    spatial_init = forwardProjectGPU_discrete( H, sample_3D, zeroImageEx_crop, exsize_crop, unique_array, max_PSF_size); % one use H and one use Ht   
    spatial_init = gather(spatial_init);
% 	figure(202), imagesc(spatial_init_debug), axis equal, axis off, title(sprintf('component %d', i))
    
    % spatial mask initialization
%     seg_mask= gather(forwardFUN(sample_mask_3D));
    sample_mask_3D = double(sample_mask_3D > 0);
    seg_mask = forwardProjectGPU_discrete(H, sample_mask_3D, zeroImageEx_crop, exsize_crop, unique_array, max_PSF_size); % one use H and one use Ht   
    seg_mask = gather(seg_mask);
    seg_mask = seg_mask > max(seg_mask(:)) * 0.1;
    
    if i == 1
        subplot(1, 2, 1);
        h_ax1 = imagesc(spatial_init);
        title(sprintf('component %d', i));
        axis equal;
        axis off;
        subplot(1, 2, 2);
        h_ax2 = imagesc(seg_mask);
        axis equal;
        axis off;
        title(sprintf('mask %d', i));
    else
        h_ax1.CData = spatial_init;
        h_ax2.CData = seg_mask;
        h_ax1.Parent(1).Title.String = sprintf('Component %d', i);
        h_ax2.Parent(1).Title.String = sprintf('Mask %d', i);
    end
    
    % temporal activity initialization
    sig_raw = squeeze(sum(bsxfun(@times, spatial_init / sum(spatial_init(:))...
        , sensor_movie_de_bg(top_left_corner(1) : bottom_right_corner(1), ...
        top_left_corner(2) : bottom_right_corner(2), :)), [1, 2]));   

    % judge if this segments contains enough activity    
    temporal_init = sig_raw;

    % record
    S_init{component_num} = single(spatial_init);
    T_init{component_num} = single(temporal_init);
    S_mask_init{component_num} = logical(seg_mask);
    bias_init{component_num} = [top_left_corner(:).'; bottom_right_corner(:).']; % rows for two corners
    component_num = component_num + 1;
end

%% S and T initialization, in case that there is empty records
S = [];
T = [];
S_mask = [];
valid_ind = 1;
for i = 1 : length(S_init)
    temp_S = S_init{i};
    temp_T = T_init{i};
    if isempty(temp_S) || isempty(temp_T)
        continue
    else
        S{valid_ind} = temp_S;
        T{valid_ind} = temp_T;
        S_mask{valid_ind} = S_mask_init{i};
        bias{valid_ind} =  bias_init{i};
        valid_ind = valid_ind + 1;
    end
end

