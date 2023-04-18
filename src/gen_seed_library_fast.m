function [S_init, T_init, S_mask_init] = gen_seed_library_fast(PSF_struct, recon_stack, valid_seg, ...
    sensor_movie, bg_spatial, bg_temporal, spike_diff_threshold, outdir, gpu_device)
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
%       seed_iter_lib: cell array, contains grascale mask for each segemnts
%           and a binary mask 
%           note for each individual patch inside segments, the library is
%           stand alone for temporal activity test? it might costs too much
%           source


%  update:
%       save the split component in target folder, with colorful label
%       for the component check: use skip recon. after check, use full
%       recon
%  to do:
%       restrict the spatial range of the component, improve the library
%       recon speed
%       discard the segments if the temporal activities are not high
%       enougth

%  last update: 3/20/2020. YZ

%% configuration
% calcium signals
gama = 0.9;
lam = 0.1;
maxIter = 100;
decimate = 1;

% LFM
recon_step = 5;
H = PSF_struct.H;
Ht = PSF_struct.Ht;
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
    
    %backwardFUN = @(projection) backwardProjectGPU(Ht, projection, zeroImageEx, exsize);
    forwardFUN = @(Xguess) forwardProjectGPU( H, Xguess, zeroImageEx, exsize); % one use H and one use Ht
    %forwardFUN_step = @(Xguess, z_depth) forwardProjectGPU_skip_2D( H, Xguess, zeroImageEx, exsize, recon_step, z_depth);
else
	forwardFUN =  @(Xguess) forwardProjectACC( H, Xguess, CAindex ); % build the function: forward and backward
    %backwardFUN = @(projection) backwardProjectACC(Ht, projection, CAindex ); 
end

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
hfig = figure();
% segment-by-segment loop
for i = 1 : N_seg % loop 1
    
    fprintf('\t build component No.%d...  \n', i)
    % read all patches and positions from valid_seg
    curr_seg = valid_seg{i, 1}; % still a cell array
    curr_pos = valid_seg{i, 2}; % a matrix
    num_comp_in_seg = length(curr_seg);
    patch_size = size(curr_seg{1}, 1);
    % no split       
    sample_3D = zeros(size_h, size_w, size_z);
    sample_mask_3D = zeros(size_h, size_w, size_z);
    for kk = 1 : num_comp_in_seg
        sample_3D(curr_pos(kk, 1) - floor(patch_size / 2) : curr_pos(kk, 1) + floor(patch_size / 2), ...
          curr_pos(kk, 2)  - floor(patch_size / 2) : curr_pos(kk, 2)+ floor(patch_size / 2), ...
          curr_pos(kk, 3) ) = ...
          sample_3D(curr_pos(kk, 1) - floor(patch_size / 2) : curr_pos(kk, 1) + floor(patch_size / 2), ...
          curr_pos(kk, 2)  - floor(patch_size / 2) : curr_pos(kk, 2)+ floor(patch_size / 2), ...
          curr_pos(kk, 3) ) + curr_seg{kk};   

        sample_mask_3D(curr_pos(kk, 1) - floor(patch_size / 2) : curr_pos(kk, 1) + floor(patch_size / 2), ...
          curr_pos(kk, 2)  - floor(patch_size / 2) : curr_pos(kk, 2)+ floor(patch_size / 2), ...
          curr_pos(kk, 3) ) = 1;  
    end         
    % spatial initialization
    spatial_init = gather(forwardFUN(sample_3D));
    if i == 1
        hax = imagesc(spatial_init);
        axis equal;
        axis off;
        colorbar;
    else
        hax.CData = spatial_init;
    end
    title(sprintf('Component %d', i));
        
    % spatial mask initialization
    seg_mask= gather(forwardFUN(sample_mask_3D));
    seg_mask = seg_mask > max(seg_mask(:)) * 0.2;

    % temporal activity initialization
    sig_raw = squeeze(sum(bsxfun(@times, spatial_init / sum(spatial_init(:))...
        , sensor_movie_de_bg), [1, 2]));   

    % judge if this segments contains enough activity    
    temporal_init = sig_raw;

    % record
    S_init{component_num} = single(spatial_init);
    T_init{component_num} = single(temporal_init);
    S_mask_init{component_num} = logical(seg_mask);

    component_num = component_num + 1;
end

%% S and T matrix initialization
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
        valid_ind = valid_ind + 1;
    end
end
S_init = S;
T_init = T;
S_mask_init = S_mask;
