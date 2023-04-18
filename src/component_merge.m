function [merged_S, merged_bias, merged_T, merged_T_deconv, merged_spike, merged_seg] =...
        component_merge(S, bias, neuron_trace_mat,  deconvol_neuron_trace_mat, spike_mat, valid_seg, ...
        movie_size, SI)
%% merge component based on spatial similarity and temporal correlations
%  last update: 12/25/2020. YZ


%% parser
% temporal_merging_threshold = 0.6; % larger than this will be merged.
% spatial_simi_threhold = 0.15; % larger than this will be merged
temporal_merging_threshold = SI.temporal_merging_threshold;
spatial_simi_threhold = SI.spatial_simi_threhold;


D_spatial = 1 - seed_pdist(S, bias, movie_size);
D_spatial = D_spatial(:).';
Z_spatial = linkage(D_spatial, 'single');

% cut based on the neuron size
T_spatial = cluster(Z_spatial, 'cutoff', 1 - spatial_simi_threhold, 'criterion', 'distance');
N_spatial_seg = max(T_spatial);

% initialization
merged_S = {};
merged_bias = {};

merged_T = [];
merged_T_deconv = [];
merged_spike = [];

merged_seg = {};

for j = 1 : N_spatial_seg % for all segments 
    % get ind for subgroup
	ind_spatial = find(T_spatial == j);  % spatial group
    
    % in case there is only 1 component
    if length(ind_spatial) == 1
        % direct record
        % S
        merged_S = [merged_S, S(ind_spatial)];
        
        % bias
        merged_bias = [merged_bias, bias(ind_spatial)];
        
        % T, deconv T and spike
        merged_T = [merged_T;  neuron_trace_mat(ind_spatial, :)];
        merged_T_deconv = [merged_T_deconv; deconvol_neuron_trace_mat(ind_spatial, :)];
        merged_spike = [merged_spike; spike_mat(ind_spatial, :)];        
        
        % seg
        merged_seg = [merged_seg; valid_seg(ind_spatial, :)];
        
    % multiple components, run an additional temporal clustering
    else
        % spatial and temporal subgroup
        activity_subgroup = neuron_trace_mat(ind_spatial, :);

        D_temporal = pdist(activity_subgroup,'correlation'); % one minus correlation value
        Z_temporal = linkage(D_temporal, 'single');
        T_temporal = cluster(Z_temporal, 'cutoff', 1 - temporal_merging_threshold, 'criterion', 'distance'); 

        N_temporal_seg = max(T_temporal);

            % record and merge
        for jj = 1 : N_temporal_seg
            % run merging
            ind_temporal = find(T_temporal == jj);
            sub_subgroup = ind_spatial(ind_temporal);

            % merge T
            buff_T = neuron_trace_mat(sub_subgroup, :);
            merged_T = [merged_T; mean(buff_T, 1)];

            buff_T_deconv = deconvol_neuron_trace_mat(sub_subgroup, :);
            merged_T_deconv = [merged_T_deconv; mean(buff_T_deconv, 1)];

            buff_spike = spike_mat(sub_subgroup, :);
            merged_spike = [merged_spike; mean(buff_spike, 1)];

            % merge seg, directly append
            buff_seg = valid_seg(sub_subgroup, :);
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
             merged_seg = [merged_seg; new_seg];

            % for those components with stand alone activity
            if length(sub_subgroup) == 1
            % S
            merged_S = [merged_S, S(sub_subgroup)];

            % bias
            merged_bias = [merged_bias, bias(sub_subgroup)];
            else
            % merge S, note we have to consider the bias. require
            % length(sub_subgroup) >=2
            temp_avg_power = 0;
            buff_S = S(sub_subgroup);
            buff_bias = bias(sub_subgroup);
            new_S = zeros(length(buff_S), movie_size(1), movie_size(2)); % hard code here. just save some time
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
            [int_array_sub_h, int_array_sub_w] = ind2sub([movie_size(1), movie_size(2)], int_array);
            new_bias_top = min(int_array_sub_h);
            new_bias_left = min(int_array_sub_w);
            new_bias_bottom = max(int_array_sub_h);
            new_bias_right = max(int_array_sub_w);
%             [new_bias_top, new_bias_left]= ind2sub([movie_size(1), movie_size(2)], int_array(1));
%             [new_bias_bottom, new_bias_right]= ind2sub([movie_size(1), movie_size(2)], int_array(end));
            new_bias = [new_bias_top, new_bias_left; new_bias_bottom, new_bias_right];
            new_S_cut = new_S(new_bias(1, 1) : new_bias(2, 1), new_bias(1, 2) : new_bias(2, 2));
            % record S and bias
            merged_S = [merged_S, {new_S_cut}];
            merged_bias = [merged_bias, {new_bias}];
        end
        end
    end
end


end