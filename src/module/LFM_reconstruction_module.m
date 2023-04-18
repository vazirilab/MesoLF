function recon_stack = LFM_reconstruction_module(summary_image, H, Ht, SI)

%% module for LFM reconstruction.
%   Input
%   summary_image: summarized xy image
%   H, Ht: forward and backward projection of LFM  
%   SI: configuration structure
%       SI.recon_max_iter: iteration number of reconstruction
%       SI.gpu_ids: GPU indices (starting with 1) to use for reconstruction
%       SI.recon_mode: you can choose phase space/phase space with peeling
%       SI.vessel_mask_enable: choose whether filter the blood vessel or
%       not
%   Output
%   recon_stack: reconstructed 3D volume.

%   last update: 5/31/2020. YZ

%% parser
iter = SI.recon_max_iter;
gpuFlag = numel(SI.gpu_ids) > 0;
recon_mode = SI.recon_mode;
vessel_mask_enable = SI.vessel_mask_enable;

%% reconstruction
disp('3D reconstruction')
if gpuFlag
    try
        gpu_id = SI.gpu_ids(mod(SI.worker_ix, numel(SI.gpu_ids)) + 1);
    catch
        gpu_id = SI.gpu_ids(1);
    end
end
if strcmp(recon_mode, 'phase_space_peeling')
    recon_stack = recon_theory_phase_space_LFM_with_filter_simple_peeling(summary_image, ...
        H, Ht, iter, gpuFlag, gpu_id);
elseif strcmp(recon_mode, 'phase_space')
    recon_stack = recon_theory_phase_space_LFM_with_filter(summary_image, ...
        H, Ht, iter, gpuFlag, gpu_id);
else
    error('Unsupported value of SI.recon_mode');
end

% balence top layer intensity
recon_stack(:, :, 1) = recon_stack(:, :, 1) * max(recon_stack(:, :, 2), [], 'all') / max(recon_stack(:, :, 1), [], 'all');

% adjust contrast
for i = 1 : size(recon_stack, 3)
    layer = recon_stack(:, :, i);
    layer = layer / max(layer(:));
    recon_stack(:, :, i) = layer;
end

%% post process and save
% apply blood vessel mask
if vessel_mask_enable
    recon_stack  = filter_with_bloodvessel_mask(recon_stack, SI);
end

recon_stack = gather(recon_stack);
recon_stack = recon_stack / max(recon_stack(:));
saveastiff(im2uint16(recon_stack / max(recon_stack(:))),...
    fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_summary_img_recon_with_' SI.detrend_mode '_detrend.tif']));

