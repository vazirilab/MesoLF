function [S, T, S_mask, S_ring, T_ring, S_ring_mask, bias] = gen_seed_library_local_with_ring_bg_same_size(PSF_struct, valid_seg, ...
    sensor_movie, recon, bg_spatial, bg_temporal, ring_ratio, gpu_ids, psf_cache_dir)
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
%       to enable GPU parallel computing, force the elements to have the
%       same size, with an additional cropping info.

%  last update: 8/27/2020. YZ

%% configuration

% LFM
H = PSF_struct.H;
Nnum = size(H ,3);
[size_h, size_w] = size(bg_spatial);

if size(bg_spatial, 3) > 1 || size(bg_temporal, 1) > 1
    error('bg_spatial and bg_temporal have to be rank-1');
end

N_seg = size(valid_seg, 1);

% if gpu_devices ~= false
%     xsize = [size_h, size_w];
%     msize = [size(H,1), size(H,2)];
%     mmid = floor(msize/2);
%     exsize = xsize + mmid;  % to make the size as 2^N after padding
%     exsize = [ min( 2^ceil(log2(exsize(1))), 128*ceil(exsize(1)/128) ), min( 2^ceil(log2(exsize(2))), 128*ceil(exsize(2)/128) ) ];
%     zeroImageEx = gpuArray(zeros(exsize, 'single'));
%
%     backwardFUN = @(projection) backwardProjectGPU(Ht, projection, zeroImageEx, exsize);
%     forwardFUN = @(Xguess) forwardProjectGPU(H, Xguess, zeroImageEx, exsize); % one use H and one use Ht
%     forwardFUN_step = @(Xguess, z_depth) forwardProjectGPU_skip_2D(H, Xguess, zeroImageEx, exsize, recon_step, z_depth);
% else
% 	forwardFUN = @(Xguess) forwardProjectACC(H, Xguess, CAindex); % build the function: forward and backward
%     backwardFUN = @(projection) backwardProjectACC(Ht, projection, CAindex);
% end

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

%% main loop
disp('Building forward components')
S_init = cell(N_seg, 1);
T_init = cell(N_seg, 1);
S_mask_init = cell(N_seg, 1);
S_ring_init = cell(N_seg, 1);
T_ring_init = cell(N_seg, 1);
S_ring_mask_init = cell(N_seg, 1);
bias_init = cell(N_seg, 1);

% if gpu_ids ~= false
%     gpu = gpuDevice(gpu_ids(1));
%     n_workers_per_gpu = floor(0.5 * gpu.TotalMemory / whos('H').bytes);
%     n_workers = numel(gpu_ids) * n_workers_per_gpu;
%     [~, rand_string] = fileparts(tempname());
%     psf_mmap_fn = fullfile(psf_cache_dir, ['sid_psf_' rand_string '.mat']);
%     save(psf_mmap_fn, 'H', '-v7.3', '-nocompression');
%     mmap = matfile(psf_mmap_fn);
% else
%     n_workers_per_gpu = 1;
%     n_workers = gcp().NumWorkers;
% end
% 
% if gpu_ids ~= false
%     n_wait = 0;
%     while (gpu.AvailableMemory < (gpu.TotalMemory * 0.6)) && (n_wait < 15)
%         disp('Pausing 60 s to wait for GPU to become available ');
%         pause(60);
%         n_wait = n_wait + 1;
%     end
% end

% prepare the last size
curr_valid_seg = valid_seg(1,:);  % avoid broadcasting entire valid_seg 2D cell array
curr_seg = curr_valid_seg{1}; % still a cell array
patch_size = size(curr_seg{1}, 1);
max_size = 5 *  patch_size + size(H, 1);


tic;
% cluster = parcluster();
% n_workers_sav = cluster.NumWorkers;
% cluster.NumWorkers = n_workers;
% cluster.NumThreads = 4;
% delete(gcp('nocreate'));
% parpool(n_workers);
%parfor (i = 1 : N_seg, n_workers)

% if gpu_ids ~= false
%                 gpu_ix = mod(i, numel(gpu_ids)) + 1;
%                 gpuDevice(gpu_ids(gpu_ix));
%                 Hgpu = 0;
% end

    
c_sample_3D_ext = cell(N_seg, 1);
c_sample_mask_3D_ext = cell(N_seg, 1);
c_ring_3D_ext = cell(N_seg, 1);
c_ring_mask_3D_ext = cell(N_seg, 1);
c_unique_array = cell(N_seg, 1);
c_max_PSF_size = cell(N_seg, 1);
c_size_buf = cell(N_seg, 1);

for i = 1 : N_seg
    %for i = 1 : N_seg
    fprintf('\t build component No.%d...  \n', i)
    % read all patches and positions from valid_seg
    curr_valid_seg = valid_seg(i,:);  % avoid broadcasting entire valid_seg 2D cell array
    curr_seg = curr_valid_seg{1}; % still a cell array
    curr_pos = curr_valid_seg{2}; % a matrix
    num_comp_in_seg = length(curr_seg);
    patch_size = size(curr_seg{1}, 1);
    
    % depth bin
    depth_array = curr_pos(:, 3);
    [unique_array, ~, ic] = unique(depth_array);
    
    % maximum bias from NIP
    [~, max_size_PSF_ind] = max(abs(depth_array - size(H, 5) / 2));
    max_PSF_size = H_range(depth_array(max_size_PSF_ind));
    
    % segnents lateral range estimation
    tl(1) = min(curr_pos(:, 1));
    tl(2) = min(curr_pos(:, 2));
    br(1) = max(curr_pos(:, 1));
    br(2) = max(curr_pos(:, 2));
    
    [r_shift, c_shift] = get_nhood(round(patch_size * ring_ratio), 1);
    [r_shift_mask, c_shift_mask] = get_nhood(patch_size * 2, 3);
    
    tl = round(max(tl - round(patch_size * ring_ratio) - max_PSF_size / 2, 1));
    % align with the Nnum
    tl = tl - mod(tl, Nnum) + 1;
    
    br = round(br + round(patch_size * ring_ratio) + max_PSF_size / 2);
    
    size_buf = br - tl + 1;
    size_buf = ceil(size_buf / Nnum) * Nnum;
    
    % boundary limit
    br(1) = min(tl(1) + size_buf(1) - 1, size_h);
    br(2) = min(tl(2) + size_buf(2) - 1, size_w);
    size_buf = br - tl + 1;
    
    % safe boundary
    size_buf(1) = min(size_buf(1), max_size);
    size_buf(2) = min(size_buf(2), max_size);
    br = tl + size_buf - 1;
    
    % build small volume
    sample_3D = zeros(size_buf(1), size_buf(2), length(unique_array));
    sample_mask_3D = zeros(size_buf(1), size_buf(2), length(unique_array));
    ring_3D = zeros(size_buf(1), size_buf(2),length(unique_array));
    ring_mask_3D= zeros(size_buf(1), size_buf(2), length(unique_array));
    
    % collect different segments
    for kk = 1 : num_comp_in_seg
        buf2 = zeros(size_h, size_w);
        buf2(curr_pos(kk, 1) - floor(patch_size / 2) : curr_pos(kk, 1) + floor(patch_size / 2), ...
            curr_pos(kk, 2)  - floor(patch_size / 2) : curr_pos(kk, 2)+ floor(patch_size / 2)) =  curr_seg{kk};
        sample_3D(:, :, ic(kk)) = sample_3D(:, :, ic(kk)) + ...
            buf2(tl(1) : br(1), tl(2) : br(2));
        
        buf2 = zeros(size_h, size_w);
        buf2(curr_pos(kk, 1) - floor(patch_size / 2) : curr_pos(kk, 1) + floor(patch_size / 2), ...
            curr_pos(kk, 2)  - floor(patch_size / 2) : curr_pos(kk, 2)+ floor(patch_size / 2)) = 1;
        sample_mask_3D(:, :, ic(kk)) = sample_mask_3D(:, :, ic(kk)) + ...
            buf2(tl(1) : br(1), tl(2) : br(2));
        
        % outer ring mask
        buf2 = zeros(size_h, size_w);
        for kkk = 1 : length(r_shift_mask)
            buf2(max(min(curr_pos(kk, 1) + r_shift_mask(kkk), size_h), 1), ... % slightly larger
                max(min(curr_pos(kk, 2) + c_shift_mask(kkk), size_w), 1)) = 1;
        end
        ring_mask_3D(:, :, ic(kk)) = ring_mask_3D(:, :, ic(kk)) + ...
            buf2(tl(1) : br(1), tl(2) : br(2));
        
        % outer ring
        buf2 = zeros(size_h, size_w);
        for kkk = 1 : length(r_shift)
            buf2(max(min(curr_pos(kk, 1) + r_shift(kkk), size_h), 1), ... % slightly larger
                max(min(curr_pos(kk, 2) + c_shift(kkk), size_w), 1)) = 1;
        end
        buf3 = buf2 .* recon(:, :, depth_array(kk)); % get intensity from reconstruction
        ring_3D(:, :, ic(kk)) = ring_3D(:, :, ic(kk)) + ...
            buf3(tl(1) : br(1), tl(2) : br(2));
        
    end
    % embed the small volume to a normal larger
    sample_3D_ext = zeros(max_size,max_size, length(unique_array));
    sample_3D_ext(1 : size_buf(1), 1 : size_buf(2), :) = sample_3D;
    sample_mask_3D_ext  = zeros(max_size,max_size, length(unique_array));
    sample_mask_3D_ext(1 : size_buf(1), 1 : size_buf(2), :) = sample_mask_3D;
    ring_3D_ext  = zeros(max_size,max_size,length(unique_array));
    ring_3D_ext(1 : size_buf(1), 1 : size_buf(2), :) = ring_3D;
    ring_mask_3D_ext = zeros(max_size,max_size, length(unique_array));
    ring_mask_3D_ext(1 : size_buf(1), 1 : size_buf(2), :) = ring_mask_3D;
    
    % Padding to make FFT input sizes equal to power of two. Not needed
    % anymore for cuda convolution, which doesn't use FFT
%     xsize_crop = [size(H, 1), size(H, 2)];
%     msize_crop = [size(H, 1), size(H, 2)];
%     mmid_crop = floor(msize_crop/2);
%     exsize_crop = xsize_crop + mmid_crop;  % to make the size as 2^N after padding
%     exsize_crop = [ min( 2^ceil(log2(exsize_crop(1))), 128*ceil(exsize_crop(1)/128) ),...
%         min( 2^ceil(log2(exsize_crop(2))), 128*ceil(exsize_crop(2)/128) ) ];
%     zeroImageEx_crop = zeros(exsize_crop, 'single');

    exsize_crop = size(sample_3D_ext, [1 2]);
    
    c_sample_3D_ext{i} = sample_3D_ext;
    c_sample_mask_3D_ext{i} = sample_mask_3D_ext;
    c_ring_3D_ext{i} = ring_3D_ext;
    c_ring_mask_3D_ext{i} = ring_mask_3D_ext;
    c_unique_array{i} = unique_array;
    c_max_PSF_size{i} = max_PSF_size;
    c_size_buf{i} = size_buf;
    bias_init{i} = [tl(:).'; br(:).']; % rows for two corners
end
toc

%% Collect and fw-project
gpu = gpuDevice(gpu_ids(1));
reset(gpu);
m_spatial_init_ext = gpuArray(zeros([exsize_crop N_seg], 'single'));
m_seg_mask_ext = gpuArray(zeros([exsize_crop N_seg], 'single'));
m_ring_init_ext = gpuArray(zeros([exsize_crop N_seg], 'single'));
m_ring_mask_ext = gpuArray(zeros([exsize_crop N_seg], 'single'));
tic;
for zi = 1 : size(H,5)    
    % from all the segments, collect all the slices that contain the
    % current z plane
    curr_slice_ixs = cellfun(@(x)(find(x == zi)), c_unique_array, 'UniformOutput', false);
    curr_slices_sample = zeros([exsize_crop nnz([curr_slice_ixs{:}])], 'single');
    curr_slices_sample_mask = zeros([exsize_crop nnz([curr_slice_ixs{:}])], 'single');
    curr_slices_ring = zeros([exsize_crop nnz([curr_slice_ixs{:}])], 'single');
    curr_slices_ring_mask = zeros([exsize_crop nnz([curr_slice_ixs{:}])], 'single');
    segs_with_curr_slice = find(~cellfun(@isempty, curr_slice_ixs));
    ptr = 1;
    for i = 1 : size(curr_slice_ixs,1)
        if ~isempty(curr_slice_ixs{i})
            curr_slices_sample(:,:,ptr) = c_sample_3D_ext{i}(:,:,curr_slice_ixs{i});
            curr_slices_sample_mask(:,:,ptr) = c_sample_mask_3D_ext{i}(:,:,curr_slice_ixs{i}) > 0;
            curr_slices_ring(:,:,ptr) = c_ring_3D_ext{i}(:,:,curr_slice_ixs{i});
            curr_slices_ring_mask(:,:,ptr) = c_ring_mask_3D_ext{i}(:,:,curr_slice_ixs{i}) > 0;
            ptr = ptr + 1;
        end
    end
    % project them all forward
    % CAUTION: to use the cuda convolution implementation as a forward projection, 
    % we have to give it the transposed PSF, usually called Ht. This is because the cuda
    % convolution actually implements a backwards projection, so to use it
    % as a forward projection we have to give it the transposed PSF!
    psf_slice = gpuArray(squeeze(single(PSF_struct.Ht(:,:,:,:,zi))));
    
    if segs_with_curr_slice > 0
        vols_slice = gpuArray(curr_slices_sample);
        proj = mex_lfm_convolution_vec(vols_slice, psf_slice);
        % add them to the accumulated projections
        for i = 1:numel(segs_with_curr_slice)
            m_spatial_init_ext(:,:,segs_with_curr_slice(i)) = m_spatial_init_ext(:,:,segs_with_curr_slice(i)) + proj(:,:,i);
        end

        vols_slice = gpuArray(curr_slices_sample_mask);
        proj = mex_lfm_convolution_vec(vols_slice, psf_slice);
        % add them to the accumulated projections
        for i = 1:numel(segs_with_curr_slice)
            m_seg_mask_ext(:,:,segs_with_curr_slice(i)) = m_seg_mask_ext(:,:,segs_with_curr_slice(i)) + proj(:,:,i);
        end

        vols_slice = gpuArray(curr_slices_ring);
        proj = mex_lfm_convolution_vec(vols_slice, psf_slice);
        % add them to the accumulated projections
        for i = 1:numel(segs_with_curr_slice)
            m_ring_init_ext(:,:,segs_with_curr_slice(i)) = m_ring_init_ext(:,:,segs_with_curr_slice(i)) + proj(:,:,i);
        end

        vols_slice = gpuArray(curr_slices_ring_mask);
        proj = mex_lfm_convolution_vec(vols_slice, psf_slice);
        % add them to the accumulated projections
        for i = 1:numel(segs_with_curr_slice)
            m_ring_mask_ext(:,:,segs_with_curr_slice(i)) = m_ring_mask_ext(:,:,segs_with_curr_slice(i)) + proj(:,:,i);
        end
    end
end
m_spatial_init_ext = gather(m_spatial_init_ext);
m_seg_mask_ext = gather(m_seg_mask_ext);
m_ring_init_ext = gather(m_ring_init_ext);
m_ring_mask_ext = gather(m_ring_mask_ext);
toc
gpuDevice([]);

%% post-process fw-projections and crop
for i = 1 : N_seg
    %disp(i)
    spatial_init_ext  = squeeze(double(m_spatial_init_ext(:,:,i)));
    
    seg_mask_ext = squeeze(double(m_seg_mask_ext(:,:,i)));
    seg_mask_ext = seg_mask_ext > max(seg_mask_ext(:)) * 0.1;
    
    ring_init_ext = squeeze(double(m_ring_init_ext(:,:,i)));
    
    ring_mask_ext = squeeze(double(m_ring_mask_ext(:,:,i)));
    ring_mask_ext = ring_mask_ext > max(ring_mask_ext(:)) * 0.1;
    
    size_buf = c_size_buf{i};
    
    % crop back and gather
    S_init{i} = single(spatial_init_ext(1 : size_buf(1), 1 : size_buf(2)));
    S_mask_init{i} = logical(seg_mask_ext(1 : size_buf(1), 1 : size_buf(2)));
    S_ring_init{i} = single(ring_init_ext(1 : size_buf(1), 1 : size_buf(2)));
    S_ring_mask_init{i} = logical(ring_mask_ext(1 : size_buf(1), 1 : size_buf(2)));
end

%     if i == 1
%         subplot(2, 2, 1);
%         h_ax1 = imagesc(spatial_init);
%         title(sprintf('component %d', i));
%         axis equal;
%         axis off;
%         subplot(2, 2, 2);
%         h_ax2 = imagesc(seg_mask);
%         axis equal;
%         axis off;
%         title(sprintf('mask %d', i));
%         subplot(2, 2, 3);
%         h_ax3 = imagesc(ring_init);
%         axis equal;
%         axis off;
%         title(sprintf('Ring %d', i));
%         subplot(2, 2, 4);
%         h_ax4 = imagesc(ring_mask);
%         axis equal;
%         axis off;
%         title(sprintf('Ring mask %d', i));
%     else
%         h_ax1.CData = spatial_init;
%         h_ax2.CData = seg_mask;
%         h_ax3.CData = ring_init;
%         h_ax4.CData = ring_mask;
%         h_ax1.Parent(1).Title.String = sprintf('Component %d', i);
%         h_ax2.Parent(1).Title.String = sprintf('Mask %d', i);
%         h_ax3.Parent(1).Title.String = sprintf('Ring %d', i);
%         h_ax4.Parent(1).Title.String = sprintf('Ring mask %d', i);
%     end


%toc
%delete(gcp('nocreate'));
%cluster.NumWorkers = n_workers_sav;
% clear mmap;
% delete(psf_mmap_fn);

%% Temporal activity initialization
disp('Initializing temporal components')
tic;
% detrend. The following old code duplicates sensor_movie and is thus memory heavy
%sensor_movie_de_bg = sensor_movie - reshape(bg_spatial, [], 1) * bg_temporal;
%sensor_movie_de_bg = reshape(sensor_movie_de_bg, size_h, size_w, []);

% reshaping won't duplicate memory
sensor_movie = reshape(sensor_movie, size_h, size_w, []);
for i = 1 : N_seg
    spatial_init = S_init{i};
    ring_init = S_ring_init{i};
    tl = bias_init{i}(1, :);
    br = bias_init{i}(2, :);
    bg_patch = bg_spatial(tl(1) : br(1), tl(2) : br(2));
    bg_patch_sz = size(bg_patch);
    bg_patch = reshape(bg_patch, [], 1);
    
    % element-wise multiplication of spatial component with detrended
    % sensor movie, to get initial temporal component
    sig_raw = sum(...
        bsxfun(@times, ...
        spatial_init / sum(spatial_init(:)), ...
        sensor_movie(tl(1) : br(1), tl(2) : br(2), :) - reshape(bg_patch * bg_temporal, bg_patch_sz(1), bg_patch_sz(2), [])), ...
        [1, 2]);
    % same for surroundings
    sig_ring_raw = sum(...
        bsxfun(@times, ...
        ring_init / sum(ring_init(:)), ...
        sensor_movie(tl(1) : br(1), tl(2) : br(2), :) - reshape(bg_patch * bg_temporal, bg_patch_sz(1), bg_patch_sz(2), [])), ...
        [1, 2]);
    
    % TODO: judge if this segments contains enough activity
    T_init{i} = single(squeeze(sig_raw));
    T_ring_init{i} = single(squeeze(sig_ring_raw));
end
%clear sensor_movie_de_bg;
toc

%% S and T initialization, in case that there is empty records
S = [];
T = [];
S_mask = [];

S_ring = [];
T_ring = [];
S_ring_mask = [];

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
        
        S_ring{valid_ind} = S_ring_init{i};
        T_ring{valid_ind} = T_ring_init{i};
        S_ring_mask{valid_ind} = S_ring_mask_init{i};
        
        bias{valid_ind} =  bias_init{i};
        valid_ind = valid_ind + 1;
    end
end
end
