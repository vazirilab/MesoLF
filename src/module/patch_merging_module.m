function [global_S, global_bias, global_bg_S, global_bg_T, global_T, global_spike, ...
                global_T_deconv, global_seg] = patch_merging_module(patch_info_array, SI)
%% module for merging all the component
%   Input
%       patch_info_array: array that contains patch infomation
%       SI: configuration structure
%           SI.optical_psf_ratio: voxel aspect ratio
%           SI.spatial_merging_threshold: merging threshold in pixel
%            SI.temporal_merging_threshold: merging threshould regard
%            1 - correlation
%           SI.margin_dis_threshold: pixel value which is used to judge if
%           the neuron is in the boundary
%
%   Ouput:
%      global_S: recorded LFM footprint for all neurons
%      global_bias: recoded footprint location
%      global_bg_S, global_bg_T: recorded gloable background for different
%      patchtes
%      global_T, global_T_deconv: recorded global temporal component
%      global_spike: recorded global spike event
%      global_seg: recorded global neuron segments

%  last update: 9/24/2020. YZ

%% parser
optical_psf_ratio = SI.optical_psf_ratio;
spatial_merging_threshold = SI.spatial_merging_threshold;
temporal_merging_threshold = SI.temporal_merging_threshold;
margin_dis_threshold = SI.margin_dis_threshold;
%% initialize
global_S = {};
global_bias = {};

global_bg_S = {};
global_bg_T = [];

global_T = [];
global_T_deconv = [];
global_spike = [];

global_seg = {};
% boundary
boundary_S = {};
boundary_bias = {};
boundary_neuron_seg = {};


boundary_T = [];
boundary_T_deconv = [];
boundary_spike = [];
% one by one merging

%% step 1, record component which is away from boundary
for patch_ind = 1 : length(patch_info_array)
    disp(['Patch index: ' num2str(patch_ind)]);
    %% load patch 
    % results from main demixing
    patch_dir = sprintf('%s%spatch_%02d', SI.outdir, filesep(), patch_ind);
%     dinfo = dir(fullfile(patch_dir, '*_final_components.mat'));
    dinfo = dir(fullfile(patch_dir, '*_after_merge.mat'));
    fns = {dinfo.name};
    [~,idx] = sort(fns);
    dinfo = dinfo(idx);
    thisfilename = dinfo(end).name;  
	curr_patch_result = importdata(fullfile(patch_dir, thisfilename));
    S = curr_patch_result.merged_S; S = S(:).';
    bias = curr_patch_result.merged_bias; bias = bias(:).';
    bg_spatial = curr_patch_result.bg_spatial;
    bg_temporal = curr_patch_result.bg_temporal;
    
    % temporal result
% 	dinfo = dir(fullfile(patch_dir, '*_final_temporal_result.mat'));
%     thisfilename = dinfo(end).name;  
%     curr_patch_temporal_result = importdata(fullfile(patch_dir, thisfilename));
    neuron_trace_mat = curr_patch_result.merged_T;
    deconvol_neuron_trace_mat = curr_patch_result.merged_T_deconv;
    spike_mat = curr_patch_result.merged_spike; 
    
	% neuron position
%     dinfo = dir(fullfile(patch_dir, '*_valid_segments.mat'));
%     thisfilename = dinfo(end).name; 
    curr_seg = curr_patch_result.merged_seg;
    curr_neuron_position = get_neuron_center_from_seg(curr_seg, optical_psf_ratio);
    
    % patch info
	dinfo = dir(fullfile(patch_dir, '*_patch_info.mat'));
    thisfilename = dinfo(end).name;  
    curr_patch_info = importdata(fullfile(patch_dir, thisfilename));
    curr_patch_loc = curr_patch_info.location; %[top_left; bottom_right];
    curr_patch_size = curr_patch_info.size; % [patch_size_h, patch_size_w];
    
    % divid neurons into inner neurons and marginal neurons
	[neuron_boundary, neuron_inner] = find_neuron_boundary(curr_neuron_position, ...
        curr_patch_size, margin_dis_threshold);
    
    
    % apply shift
    curr_seg = apply_patch_shift(curr_seg, curr_patch_loc);
    %% record the components that are not in the boundary
    S_inner = S(neuron_inner);
    global_S = [global_S, S_inner];
    % bias, apply patch shift
    bias_inner = bias(neuron_inner);
    for i = 1 : length(bias_inner)
        curr_bias = bias_inner{i}; % should be [top_left_corner(:).'; bottom_right_corner(:).']
        curr_bias = curr_bias + [curr_patch_loc(1, :)-1; curr_patch_loc(1, :)-1]; % add top-left corner
        global_bias = [global_bias, {curr_bias}];
    end
    % bg_S and bg_T
    global_bg_S = [global_bg_S, {bg_spatial}];
    global_bg_T = [global_bg_T; bg_temporal];

    % T, deconv T and spike
    global_T = [global_T; neuron_trace_mat(neuron_inner, :)];
    global_T_deconv = [global_T_deconv; deconvol_neuron_trace_mat(neuron_inner, :)];
    global_spike = [global_spike; spike_mat(neuron_inner, :)];
                
    global_seg = [global_seg; curr_seg(neuron_inner, :)];
    %% record the components that are in the boundary
    S_boundary = S(neuron_boundary);
    boundary_S = [boundary_S, S_boundary];
    % bias, apply patch shift
    bias_boundary = bias(neuron_boundary);
    for i = 1 : length(bias_boundary)
        curr_bias = bias_boundary{i}; % should be [top_left_corner(:).'; bottom_right_corner(:).']
        curr_bias = curr_bias + [curr_patch_loc(1, :)-1; curr_patch_loc(1, :)-1];
        boundary_bias = [boundary_bias, {curr_bias}];
    end

    % T, deconv T and spike
    boundary_T = [boundary_T; neuron_trace_mat(neuron_boundary, :)];
    boundary_T_deconv = [boundary_T_deconv; deconvol_neuron_trace_mat(neuron_boundary, :)];
    boundary_spike = [boundary_spike; spike_mat(neuron_boundary, :)];    
    
    % spatial segmentation
    boundary_neuron_seg  = [boundary_neuron_seg; curr_seg(neuron_boundary, :)];
end
%% step 2, spatially merging components in the boundary based on distance and 
%  temporal correlation

% get neuron 3D position
boundary_neuron_position = get_neuron_center_from_seg(boundary_neuron_seg, optical_psf_ratio);

if ~isempty(boundary_neuron_position)
    % run spatial clustering
    D_spatial = pdist(boundary_neuron_position);
    Z_spatial = linkage(D_spatial, 'single');
    T_spatial = cluster(Z_spatial, 'cutoff', spatial_merging_threshold, 'criterion', 'distance');

    % for those close enough, run again about the temporal clustering
    N_spatial_seg = max(T_spatial);
else
    N_spatial_seg = 0;
end
for j = 1 : N_spatial_seg    
    % get ind for subgroup
	ind_spatial = find(T_spatial == j);  
    
    % in case there is only 1 component
    if length(ind_spatial) == 1
        % direct record
        
        % S
        global_S = [global_S, boundary_S(ind_spatial)];
        
        % bias
        global_bias = [global_bias, boundary_bias(ind_spatial)];
        
        % T, deconv T and spike
        global_T = [global_T; boundary_T(ind_spatial, :)];
        global_T_deconv = [global_T_deconv; boundary_T_deconv(ind_spatial, :)];
        global_spike = [global_spike; boundary_spike(ind_spatial, :)];        
        
        % seg
        global_seg = [global_seg; boundary_neuron_seg(ind_spatial, :)];
        
    % multiple components, run an additional temporal clustering
    else
        % spatial and temporal subgroup
        activity_subgroup = boundary_T(ind_spatial, :);

        D_temporal = pdist(activity_subgroup,'correlation'); % one minus correlation value
        Z_temporal = linkage(D_temporal, 'single');
        T_temporal = cluster(Z_temporal, 'cutoff', temporal_merging_threshold, 'criterion', 'distance'); 

        N_temporal_seg = max(T_temporal);
        
        % record and merge
        for jj = 1 : N_temporal_seg
            % run merging
            ind_temporal = find(T_temporal == jj);
            sub_subgroup = ind_spatial(ind_temporal);

            % merge T
            buff_T = boundary_T(sub_subgroup, :);
            global_T = [global_T; mean(buff_T, 1)];

            buff_T_deconv = boundary_T_deconv(sub_subgroup, :);
            global_T_deconv = [global_T_deconv; mean(buff_T_deconv, 1)];

            buff_spike = boundary_spike(sub_subgroup, :);
            global_spike = [global_spike; mean(buff_spike, 1)];

            % merge seg, directly append
            buff_seg = boundary_neuron_seg(sub_subgroup, :);
            temp_seg = {};
            temp_seg_pos = [];
            for kk = 1 : size(buff_seg, 1)
                temp = buff_seg{kk, 1};
                % segments record
                temp_seg = [temp_seg; temp];
                % position record
                temp_seg_pos = [temp_seg_pos; buff_seg{kk, 2}];
            end        
            new_seg = {temp_seg, temp_seg_pos};
            global_seg = [global_seg; new_seg];
            
            % for those components with stand alone activity
            if length(sub_subgroup) == 1
                % S
                global_S = [global_S, boundary_S(sub_subgroup)];

                % bias
                global_bias = [global_bias, boundary_bias(sub_subgroup)];
            else
                % merge S, note we have to consider the bias. require
                % length(sub_subgroup) >=2
                temp_avg_power = 0;
                buff_S = boundary_S(sub_subgroup);
                buff_bias = boundary_bias(sub_subgroup);
                new_S = zeros(length(buff_S), 7000, 7000); % hard code here. just save some time
                for kk = 1 : length(buff_S)
                    temp_bias = buff_bias{kk};

                    % record 
                    new_block = buff_S{kk};

                    % record energy for summation
                    temp_avg_power(kk) = max(new_block(:)) ;

                    new_S(kk, temp_bias(1,1) : temp_bias(2,1), temp_bias(1,2) : temp_bias(2,2)) ...
                        =  new_block;
                end  
                new_S = squeeze(sum(bsxfun(@rdivide, new_S, temp_avg_power(:)) / min(temp_avg_power), 1));
                % crop
                int_array = find(new_S > 0.1 * max(new_S(:)));
%                 [new_bias_top, new_bias_left]= ind2sub([7000, 7000], int_array(1));
%                 [new_bias_bottom, new_bias_right]= ind2sub([7000, 7000], int_array(end));

                [int_array_sub_h, int_array_sub_w] = ind2sub([7000, 7000], int_array);
                new_bias_top = min(int_array_sub_h);
                new_bias_left = min(int_array_sub_w);
                new_bias_bottom = max(int_array_sub_h);
                new_bias_right = max(int_array_sub_w);
                new_bias = [new_bias_top, new_bias_left; new_bias_bottom, new_bias_right];
                
                new_S_cut = new_S(new_bias(1, 1) : new_bias(2, 1), new_bias(1, 2) : new_bias(2, 2));
                % record S and bias
                global_S = [global_S, {new_S_cut}];
                global_bias = [global_bias, {new_bias}];
            end
            

        end
    end
end
end

function neuron_position = get_neuron_center_from_seg(valid_seg, optical_psf_ratio)
%% find the neuron center
% require pixel size
neuron_position = [];
for i = 1 : size(valid_seg, 1)
   pos = valid_seg{i, 2};
   pos_mean = mean(pos, 1);
   pos_mean(1) = pos_mean(1) ;
   pos_mean(2) = pos_mean(2) ;
   pos_mean(3) = pos_mean(3) * optical_psf_ratio;
   neuron_position(i, :) = pos_mean;
end
end

function [neuron_boundary, neuron_inner] = find_neuron_boundary(neuron_position, patch_size, dis_threshold)
%% find the neurons which are close to the boundary, with dis_threshold as 
%  the threshold
%  neuron_position: a Nx3 matrix for N different neurons
%  patch_size: indicating the boundary size, [sdize_h, size_w]
neuron_boundary = [];
neuron_inner = [];
for i = 1 : size(neuron_position, 1)
   if  neuron_position(i, 2)-1 >  dis_threshold && ...
       patch_size(2) - neuron_position(i, 2) > dis_threshold && ...
       neuron_position(i, 1)-1 >  dis_threshold && ...
       patch_size(1) - neuron_position(i, 1) > dis_threshold
       neuron_inner = [neuron_inner ; i];       
   else
       neuron_boundary = [neuron_boundary ; i];
   end
end

end

function curr_seg = apply_patch_shift(curr_seg, curr_patch_loc)
% apply patch shift
for i = 1 : size(curr_seg, 1)
   pos = curr_seg{i, 2};
   pos(:, 1) = pos(:, 1) + curr_patch_loc(1, 1) - 1;
   pos(:, 2) = pos(:, 2) + curr_patch_loc(1, 2) - 1;

   curr_seg{i, 2} = pos;
end
end
