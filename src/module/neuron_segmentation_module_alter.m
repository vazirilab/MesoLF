function [valid_seg, discard_seg] = neuron_segmentation_module_alter(sensor_movie, volume, SI, patch_info, mask, ext_PSF_struct)
%% module for 3D neuron reconstruction.
%   Input
%   sensor_movie: xy-t movie
%   volume: reconstructed 3D stack
%   SI: configuration structure
%       SI.neuron_number: maximum per depth neuron number
%       SI.neuron_lateral_size: lateral neuron size
%       SI.local_constrast_th: neuron/background contrast
%       SI.optical_psf_ratio : psf extend ratio
%       SI.overlap_ratio: merging ratio based on patch size
%       SI.boundary_thickness: for plot; show a box for each neuron
%       SI.discard_component_save: flag, if one will save the discard
%       component
%       SI.outdir: output directory

%   Output
%   valid_seg: cell array, record each valid component
%   discard_seg: cell array, record discarded component (as candicates of neurons)

%   last update: 4/30/2022. YZ

%% parser
neuron_number = SI.neuron_number;% maximum per depth neuron number
neuron_lateral_size = SI.neuron_lateral_size;
local_constrast_th = SI.local_constrast_th;

optical_psf_ratio = SI.optical_psf_ratio;
overlap_ratio = SI.overlap_ratio;
boundary_thickness = SI.boundary_thickness; % for plotting
discard_component_save = SI.discard_component_save;

NMF_threshold = SI.NMF_threshold;
NMF_iteration = SI.NMF_iteration;
segmentation_method = SI.segmentation_method;

outdir = SI.outdir;

d1 = size(volume, 1);
d2 = size(volume, 2);
d3 = size(volume, 3);

% FOV mask
if nargin < 4 || isempty(mask)
    patch_mask = ones(patch_info.size(1), patch_info.size(2));
else
    mask = double(mask);
    patch_mask = mask(patch_info.location(1, 1): patch_info.location(2, 1), ...
                  patch_info.location(1, 2): patch_info.location(2, 2));
end

%% Segmentation
if strcmp(segmentation_method , 'morph')
options.false_threshold = 300;
options.local_constrast_th = local_constrast_th;
options.d1 = d1;
options.d2 = d2;
options.min_threshold = 0.2;
options.gSig = neuron_lateral_size;    % width of the gaussian kernel approximating one neuron
options.gSiz = round(neuron_lateral_size * 2.5);    % average size of filter size
options.ssub = 1;   

options.min_v_search = 0.01;
options.seed_method  = 'other';
options.min_pixel = neuron_lateral_size^2 * 0.8;
options.center_psf = [];
options.spatial_constraints.connected =  true;

%% preparing
volume = volume / max(volume(:));
% volume = imadjustn(volume);

Cn_stack = zeros(d1, d2, d3);
raw_segments = cell(d3, 2);
simu_segment_volume = zeros(d1, d2, d3);
residue_voume = zeros(d1, d2, d3);

for i = 1 : d3
    fprintf('segment for depth %d \n', i)
    % input
    slice = volume(:, :, i);
    
    % pre-process
    slice = imadjust(slice);
    slice = slice/ max(slice(:));
    
    [results, center, Cn] = greedyROI_endoscope_summary_shape_aware(slice(:), ...
                                            neuron_number, options, false);
                                        
    Cn_stack(:, :, i) = Cn;
    
    
    % record raw_segments
    component_lib = {};
    for j = 1 : size(center, 1)
        buf = results.Ain(:, j);
        buf = reshape(full(buf), d1, d2);
        
        % modify patch
        center(j, 1) = max(options.gSiz + 1, center(j, 1));
        center(j, 1) = min(d1 - options.gSiz, center(j, 1));
        center(j, 2) = max(options.gSiz + 1, center(j, 2));
        center(j, 2) = min(d2 - options.gSiz, center(j, 2));  
        
        patch = buf(max(center(j, 1) - options.gSig, 1) : min(center(j, 1) + options.gSig, d1), ...
            max(center(j, 2) - options.gSig, 1) : min(center(j, 2) + options.gSig, d2));
        component_lib{j} = patch;
    end
    center_stack{i} = center;
    raw_segments{i, 1} = center; % center of segments
    raw_segments{i, 2} = component_lib; % recorded segments
    
    % record dummy volume
    simulate_segment = reshape(full(sum(results.Ain, 2)), d1, d2);
    
    simu_segment_volume(:, :, i) = simulate_segment;
    residue_voume(:, :, i) = slice - simulate_segment;
    close all
end       

%% 
saveastiff(im2uint16(simu_segment_volume/ max(simu_segment_volume(:))), ...
    fullfile(outdir, [datestr(now, 'YYmmddTHHMM') '_simu_segment_volume.tif']));
saveastiff(im2uint16(residue_voume / max(residue_voume(:))), ...
    fullfile(outdir, [datestr(now, 'YYmmddTHHMM') '_residue_voume.tiff']));
saveastiff(im2uint16(Cn_stack / max(Cn_stack(:))), ...
    fullfile(outdir, [datestr(now, 'YYmmddTHHMM') '_Cn_stack.tiff']));
save(fullfile(outdir, [datestr(now, 'YYmmddTHHMM') '_raw_segments.mat']), 'raw_segments');

%% Clustering of segments
% optical_psf_ratio = 3;
% overlap_ratio = 0.5;
% boundary_thickness = 3; % for plotting
% discard_component_save = false;

volume_threshold = [(options.gSiz)^2 * 2, ... % maximum/minimum neuron size
    (options.gSiz)^2 * 500];

valid_component_image_save = false;
[valid_seg, discard_seg] = hierarchical_clustering(raw_segments, ...
    volume, boundary_thickness, optical_psf_ratio, overlap_ratio, ...
    volume_threshold, SI.outdir, discard_component_save, valid_component_image_save);
if isempty(valid_seg) 
    return;
end

%% post segment
% select
keep_ind = ones(size(valid_seg, 1), 1);
for i = 1 : size(valid_seg, 1)
	curr_seg_pos =  valid_seg{i, 2};
	curr_seg_pos_center = mean(curr_seg_pos, 1);
    curr_seg_pos_center = round(curr_seg_pos_center );
    curr_seg_pos_center(1) = max(min(curr_seg_pos_center(1), patch_info.size(1)), 1);
    curr_seg_pos_center(2) = max(min(curr_seg_pos_center(2), patch_info.size(2)), 1);
	if patch_mask(curr_seg_pos_center(1), curr_seg_pos_center(2)) == 0  % outside of FOV
            keep_ind(i) = 0;
	end
end
% avoid keep ind to be all zeros. Safe belt
if sum(keep_ind) < 2 % to aviod
    keep_ind(1:2) = 1;
end
valid_seg = valid_seg(keep_ind > 0, :);


elseif strcmp(segmentation_method , 'NMF')
    options.gSig = neuron_lateral_size;    % width of the gaussian kernel approximating one neuron
    options.gSiz = round(neuron_lateral_size * 2.5);    % average size of filter size
    
    % binarize the volume
    volume_bw = volume > max(volume(:)) * NMF_threshold;
    % morphological Erosion
    SE = strel('sphere', options.gSig);
    volume_bw = imerode(volume_bw,SE);
    
    % connectom 
    CC = bwconncomp(volume_bw);
    
    % neuron number
    K = length(CC.PixelIdxList);
    A_raw = zeros(d1 * d2 * d3,  K);
    curr_vol = [];
    for i = 1 : K
        curr_list = CC.PixelIdxList(i);
        A_raw(curr_list{:}, i) = 1;
        
        % calculate volume size
        curr_vol(i) = sum(A_raw(:, i), 'all');
    end
    vol_size_indicator = curr_vol < (round(neuron_lateral_size * 2.5))^2 * 2;
    vol_size_indicator = find(vol_size_indicator);
    
    % throw away too small component
    A_raw(:, vol_size_indicator) = [];
    K = size(A_raw, 2);
    
    % forward propagation
    A_raw = reshape(A_raw, d1, d2, d3, []);
    A = zeros(d1 * d2, K);
    for i = 1 : K
        LFM_cap_bg = gather(forwardProjectACC(ext_PSF_struct.H, A_raw(:, :, :, i), ext_PSF_struct.CAindex));
        A(:, i) = LFM_cap_bg(:);
    end
    
    max_mask = max(sensor_movie, [], 2);
    max_mask = max_mask > max(max_mask(:)) * 0.1;
    maxIter = 10;
    C = rand(K, size(sensor_movie, 2));
    % NMF
    for i = 1 : NMF_iteration
        [C, C_raw, results_deconv, cc] = HALS_temporal(sensor_movie, A, C, maxIter, []);

        A = HALS_spatial(sensor_movie, A, C, max_mask, maxIter);
    end
    
    % throw away too dim neurons
    std_C = std(C, 0, 2);
    invalid_component = find(std_C < max(std_C) * 0.1);
    A(:, invalid_component) = [];
    
    
    
    valid_seg = [];
    for i = 1 : size(A, 2)
        curr_A = A(:, i);
        curr_A = reshape(curr_A, d1, d2);
        % reconstruction
        curr_A_raw = gather(LFM_reconstruction_module(curr_A, ext_PSF_struct.H, ext_PSF_struct.Ht, SI));
        
        % crop and output to valid_seg
        curr_A_raw_bw = curr_A_raw > 0.1 * max(curr_A_raw(:));
        
        % components with maximum size
        BW = zeros(size(curr_A_raw_bw));
        curr_CC = bwconncomp(curr_A_raw_bw);
        numPixels = cellfun(@numel, curr_CC.PixelIdxList);
        [biggest,idx] = max(numPixels);
        BW(curr_CC.PixelIdxList{idx}) = 1;
        
        max_z = max(BW, [], [1, 2]);
        max_z = find(max_z(:));
        
        center = [];
        patch_array = [];
        for j = 1 : length(max_z) % for each of valid depth
            % get current image
            curr_img = BW(:, :, max_z(j)) .*  curr_A_raw(:, :, max_z(j));
            
            % calculate center
            curr_center = com(curr_img(:), d1, d2);
            
            % grab patches
            center(j, 1) = max(options.gSiz + 1, curr_center(1));
            center(j, 1) = min(d1 - options.gSiz, curr_center(1));
            center(j, 2) = max(options.gSiz + 1, curr_center(2));
            center(j, 2) = min(d2 - options.gSiz, curr_center(2));  

            patch = curr_img(max(center(j, 1) - options.gSig, 1) : min(center(j, 1) + options.gSig, d1), ...
                max(center(j, 2) - options.gSig, 1) : min(center(j, 2) + options.gSig, d2));
            patch_array{j} = patch;
        end
        valid_seg{i, 1} = patch_array; % 2d segments
        valid_seg{i, 2} = center; % 3d positions
    end
    
    discard_seg = [];
    
elseif strcmp(segmentation_method , 'mix')
    options.gSig = neuron_lateral_size;    % width of the gaussian kernel approximating one neuron
    options.gSiz = round(neuron_lateral_size * 2.5);    % average size of filter size
    
    % binarize the volume
    volume_bw = volume > max(volume(:)) * NMF_threshold;
    % morphological Erosion
    SE = strel('sphere', options.gSig);
    volume_bw = imerode(volume_bw,SE);
    
    % connectom 
    CC = bwconncomp(volume_bw);
    
    % neuron number
    K = length(CC.PixelIdxList);
    A_raw = zeros(d1 * d2 * d3,  K);
    curr_vol = [];
    for i = 1 : K
        curr_list = CC.PixelIdxList(i);
        A_raw(curr_list{:}, i) = 1;
        
        % calculate volume size
        curr_vol(i) = sum(A_raw(:, i), 'all');
    end
    vol_size_indicator = curr_vol < (round(neuron_lateral_size * 2.5))^2 * 2;
    vol_size_indicator = find(vol_size_indicator);
    
    % throw away too small component
    A_raw(:, vol_size_indicator) = [];
    K = size(A_raw, 2);
    
    % forward propagation
    A_raw = reshape(A_raw, d1, d2, d3, []);
    A = zeros(d1 * d2, K);
    for i = 1 : K
        LFM_cap_bg = gather(forwardProjectACC(ext_PSF_struct.H, A_raw(:, :, :, i), ext_PSF_struct.CAindex));
        A(:, i) = LFM_cap_bg(:);
    end
    
    max_mask = max(sensor_movie, [], 2);
    max_mask = max_mask > max(max_mask(:)) * 0.1;
    maxIter = 10;
    C = rand(K, size(sensor_movie, 2));
    % NMF
    for i = 1 : NMF_iteration
        [C, C_raw, results_deconv, cc] = HALS_temporal(sensor_movie, A, C, maxIter, []);

        A = HALS_spatial(sensor_movie, A, C, max_mask, maxIter);
    end
    
    % throw away too dim neurons
    std_C = std(C, 0, 2);
    invalid_component = find(std_C < max(std_C) * 0.1);
    A(:, invalid_component) = [];
    
    
    % for each of NMF component, run a morph based segmentation
    valid_seg = [];
    discard_seg = [];
    for i = 1 : size(A, 2)
        curr_A = A(:, i);
        curr_A = reshape(curr_A, d1, d2);
        
        % reconstruction
        curr_A_raw = gather(LFM_reconstruction_module(curr_A, ext_PSF_struct.H, ext_PSF_struct.Ht, SI)); % size of d1 and d2 and d3
        
        
        % morphology segmentation
        Cn_stack = zeros(d1, d2, d3);
        raw_segments = cell(d3, 2);
%         simu_segment_volume = zeros(d1, d2, d3);
%         residue_voume = zeros(d1, d2, d3);
        center_stack = cell(d3, 1);
        for ii = 1 : d3
%             fprintf('segment for depth %d \n', ii)
            % input
            slice = curr_A_raw(:, :, ii);

            % pre-process
            slice = imadjust(slice);
            slice = slice/ max(slice(:));

            [results, center, Cn] = greedyROI_endoscope_summary_shape_aware(slice(:), ...
                                                    neuron_number, options, false);

            Cn_stack(:, :, ii) = Cn;


            % record raw_segments
            component_lib = {};
            for j = 1 : size(center, 1)
                buf = results.Ain(:, j);
                buf = reshape(full(buf), d1, d2);

                % modify patch
                center(j, 1) = max(options.gSiz + 1, center(j, 1));
                center(j, 1) = min(d1 - options.gSiz, center(j, 1));
                center(j, 2) = max(options.gSiz + 1, center(j, 2));
                center(j, 2) = min(d2 - options.gSiz, center(j, 2));  

                patch = buf(max(center(j, 1) - options.gSig, 1) : min(center(j, 1) + options.gSig, d1), ...
                    max(center(j, 2) - options.gSig, 1) : min(center(j, 2) + options.gSig, d2));
                component_lib{j} = patch;
            end
            center_stack{ii} = center;
            raw_segments{ii, 1} = center; % center of segments
            raw_segments{ii, 2} = component_lib; % recorded segments

            % record dummy volume
%             simulate_segment = reshape(full(sum(results.Ain, 2)), d1, d2);

%             simu_segment_volume(:, :, ii) = simulate_segment;
%             residue_voume(:, :, ii) = slice - simulate_segment;
            close all
        end       
        volume_threshold = [(options.gSiz)^2 * 2, ... % maximum/minimum neuron size
            (options.gSiz)^2 * 500];

        % hirachical clustering
        valid_component_image_save = false;
        [valid_seg_tmp, discard_seg_tmp] = hierarchical_clustering(raw_segments, ...
            curr_A_raw, boundary_thickness, optical_psf_ratio, overlap_ratio, ...
            volume_threshold, SI.outdir, discard_component_save, valid_component_image_save);
        if isempty(valid_seg_tmp) 
            return;
        end

        keep_ind = ones(size(valid_seg_tmp, 1), 1);
        for ii = 1 : size(valid_seg_tmp, 1)
            curr_seg_pos =  valid_seg_tmp{ii, 2};
            curr_seg_pos_center = mean(curr_seg_pos, 1);
            curr_seg_pos_center = round(curr_seg_pos_center );
            curr_seg_pos_center(1) = max(min(curr_seg_pos_center(1), patch_info.size(1)), 1);
            curr_seg_pos_center(2) = max(min(curr_seg_pos_center(2), patch_info.size(2)), 1);
            if patch_mask(curr_seg_pos_center(1), curr_seg_pos_center(2)) == 0  % outside of FOV
                    keep_ind(ii) = 0;
            end
        end
        % avoid keep ind to be all zeros. Safe belt
        if sum(keep_ind) < 2 % to aviod
            keep_ind(1:2) = 1;
        end
        valid_seg_tmp = valid_seg_tmp(keep_ind > 0, :);

        % record!
        valid_seg = [valid_seg; valid_seg_tmp];
        discard_seg = [discard_seg; discard_seg_tmp];
    end
    
    
    
    
else
    error('Segmentation methods should be *morph* or *NMF* or *mix*!')
end

end