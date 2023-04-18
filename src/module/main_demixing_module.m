function [S, T_raw, S_bg, T_bg, bg_spatial, bg_temporal] = ...
    main_demixing_module(sensor_movie, S_init, T_init, ...
    S_bg_init, T_bg_init, S_mask_init, S_bg_mask_init,...
     bg_spatial_init,  bg_temporal_init, bias,  patch_info, SI)
%% module for main demixing.
%   Input:
%       sensor_movie: xyt movie
%       S_init, T_init: neuron spatial/temporal component
%       S_bg_init, T_bg_init: neuropil spatial/temporal component
%       S_mask_init: neuron spatial mask
%       S_bg_mask_init: neuropil spatial mask
%       bias: only for local mode. record the spatial bias of each
%       bg_spatial_init, bg_temporal_init: background spatial/temporal
%       compoennt
%       patch_info: structure, contains patch location and patch size
%   Output:
%       S, T_raw: neuron spatial/temporal component after CNMF
%       S_bg, T_bg: neuropil spatial/temporal component after CNMF
%       bg_spatial, bg_temporal: rank-1 background spatial/temporal component after CNMF
% last update: 5/31/2020. YZ  

%% parser
S = S_init;
T_raw = T_init;
S_bg = S_bg_init;
T_bg = T_bg_init;
bg_spatial = bg_spatial_init(:);
bg_temporal = bg_temporal_init;

size_h = patch_info.size(1);
size_w = patch_info.size(2);
movie_size = [size_h, size_w];

nmf_max_iter = SI.nmf_max_iter;
nmf_max_round = SI.nmf_max_round;
lib_store_mode = SI.lib_store_mode;

%%
tic;
for iter = 1 : nmf_max_round
    fprintf('MesoLF main iteration %d/%d\n', iter, nmf_max_round)
    % spatial update
    if isempty(S_bg)   
        if strcmp(lib_store_mode, 'full')
            [S, bg_spatial] = update_spatial_lasso(sensor_movie, S, T_raw, ...
                bg_spatial, bg_temporal, S_mask_init, nmf_max_iter);
            [T_raw, bg_temporal] = update_temporal_lasso(sensor_movie, S, T_raw, ...
                bg_spatial, bg_temporal, S_mask_init, nmf_max_iter);
            S_bg = [];
            T_bg = [];
            bias = [];
        elseif strcmp(lib_store_mode, 'snippet')
            [S, bg_spatial] = update_spatial_lasso_local(sensor_movie,S, T_raw, ...
                bias, bg_spatial, bg_temporal, S_mask_init, movie_size, nmf_max_iter);         
            [T_raw, bg_temporal] = update_temporal_lasso_local(sensor_movie,S, ...
                T_raw, bias, bg_spatial, bg_temporal, S_mask_init, movie_size, nmf_max_iter);        
            S_bg = [];
            T_bg = []; 
        end
    else
        % only supported in local mode
        [S, S_bg, bg_spatial] = update_spatial_lasso_with_bg(sensor_movie, S, S_bg, T_raw, ...
            T_bg, bias, bg_spatial, bg_temporal, S_mask_init, S_bg_mask_init, movie_size, nmf_max_iter);        

        [T_raw, T_bg, bg_temporal] = update_temporal_lasso_with_bg(sensor_movie, ...
            S, S_bg, T_raw, T_bg, bias, bg_spatial, bg_temporal, S_mask_init, S_bg_mask_init, movie_size,nmf_max_iter);              
    end 
    % to do: merge based on spatial and temporal position
end
toc
end
