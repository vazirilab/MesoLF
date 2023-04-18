function [S_init, T_init, S_bg_init, T_bg_init, S_mask_init, S_bg_mask_init, bias] = ...
    seed_library_generation_module(sensor_movie, volume, valid_seg, PSF_struct, ...
    bg_spatial_init, bg_temporal_init, SI)
%% module for generating seed library.
%   Input:
%       sensor_movie: xyt movie
%       volume: reconstructed volume
%       valid_seg: cell array, record neuron segmentation
%       PSF_structure: record LFM PSF
%       bg_spatial_init, bg_temporal_init: background spatial/temporal
%       compoennt
%       SI: global config struct
%   Output:
%       S_init, T_init: neuron spatial/temporal component
%       S_bg_init, T_bg_init: neuropil spatial/temporal component
%       S_mask_init: neuron spatial mask
%       S_bg_mask_init: neuropil spatial mask
%       bias: only for local mode. record the spatial bias of each
%       component
%   update: add an additional refinement for non-overlapped seed
% last update: 7/7/2020. YZ  

%% parser
lib_store_mode = SI.lib_store_mode;
lib_gen_mode = SI.lib_gen_mode;
outdir = SI.outdir;

if strcmp(lib_gen_mode, 'bg_ring')
    try
        radius_ratio = SI.radius_ratio;
    catch
        error('for bg_ring mode, a radius ratio needs to be determined')
    end
end
%% library generation
buf = rand(2, size(sensor_movie, 2)) > 0.9;
spike_diff_threshold = spike_distance(buf(1, :), buf(2, :)) * 0.2;

% library generation
if strcmp(lib_store_mode, 'full')
    switch lib_gen_mode
        case 'ballistic'
            if numel(SI.gpu_ids) > 0
                gpu_device = gpuDevice(SI.gpu_ids(1));
                reset(gpu_device);
            else
                gpu_device = false;
            end
            [S_init, T_init, S_mask_init] = gen_seed_library_fast(PSF_struct, volume, valid_seg, ...
                sensor_movie, bg_spatial_init, bg_temporal_init, sqrt(spike_diff_threshold), outdir, gpu_device);
            S_bg_init = [];
            T_bg_init = [];
            S_bg_mask_init = [];     
            bias = [];
            if gpu_device
                gpuDevice([]);
            end
        otherwise
            warning('Invalid or missing value for SI.lig_gen_mode');
    end
elseif strcmp(lib_store_mode, 'snippet')
    switch lib_gen_mode
        case 'ballistic'
            if numel(SI.gpu_ids) > 0
                gpu_device = gpuDevice(SI.gpu_ids(1));
                reset(gpu_device);
            else
                gpu_device = false;
            end
            [S_init, T_init, S_mask_init, bias] = gen_seed_library_local(PSF_struct, valid_seg, ...
                    sensor_movie, bg_spatial_init, bg_temporal_init, gpu_device);
            S_bg_init = [];
            T_bg_init = [];
            S_bg_mask_init = [];
            if gpu_device
                gpuDevice([]);
            end
        case 'bg_ring'
        	[S_init, T_init, S_mask_init, S_bg_init, T_bg_init, S_bg_mask_init, bias] = ...
                 gen_seed_library_local_with_ring_bg_same_size(PSF_struct, valid_seg, ...
                 sensor_movie, volume, bg_spatial_init, bg_temporal_init, radius_ratio, SI.gpu_ids, SI.psf_cache_dir);    
                 % component refinement
            [S_bg_mask_init, S_bg_init] = library_refinement(valid_seg, bg_spatial_init, ...
                S_bg_mask_init, S_bg_init, S_mask_init, bias);
        otherwise
            warning('Invalid or missing value for SI.lig_gen_mode');
    end
end


%% delete component with NAN trace
keep_id = ones(length(S_bg_mask_init), 1);
for i = 1 : length(S_init)
    buf_temporal = T_init{i};
    if max(buf_temporal) == 0 || max(isnan(buf_temporal)) % delete
        keep_id(i) = 0;
    end
end
keep_id = find(keep_id);

S_init = S_init(keep_id);
T_init = T_init(keep_id);
S_mask_init = S_mask_init(keep_id);
S_bg_init = S_bg_init(keep_id);
T_bg_init = T_bg_init(keep_id);
S_bg_mask_init = S_bg_mask_init(keep_id);
bias = bias(keep_id);
end
