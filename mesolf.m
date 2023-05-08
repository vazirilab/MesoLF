function exit_code = mesolf(varargin)
% MesoLF  -- Run the MesoLF source extraction pipeline
% for light field microscopy calcium imaging in scattering tissue.
% use this file as a script by commenting out the first line (function ...) 
% and assigning SIin param struct to varargin variable as follows: 
% varargin = {SIin};

%%
disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Starting MesoLF with arguments:']);
addpath(genpath(fileparts(mfilename('fullpath'))));
if numel(varargin) == 1
    SI = mesolf_set_params(varargin);
else
    SI = mesolf_set_params(varargin{:});
end
disp(SI);
mkdir(SI.outdir);

%% Load psf
disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Loading PSF']);
ext_PSF_struct = load(SI.psffile);
SI.Nnum = ext_PSF_struct.Nnum;

%% Determine patch geometry
disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Determining patch geometry']);
patch_info_array = determine_patch_info_aug(SI);
for i = 1:length(patch_info_array)
    % using jsonencode() just to make sure that all fields get displayed in full
    disp(jsonencode(patch_info_array{i}));
end

%% Load mask file
if isa(SI.mask_file, 'char')
    mask = loadtiff(SI.mask_file);
elseif SI.mask_file == true
    [mfilepath, ~, ~] = fileparts(mfilename('fullpath'));
    mask_file_default = [mfilepath filesep() 'utility' filesep() 'outside_mask.tif'];
    mask = loadtiff(mask_file_default);
else
    mask = [];
end

%% Main loop over patches
if SI.worker_ix == 0
    my_patch_ixs = 1 : length(patch_info_array);
elseif SI.worker_ix <= length(patch_info_array)
    my_patch_ixs = [SI.worker_ix];
else
    my_patch_ixs = [];
end

for patch_ind = my_patch_ixs
    disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Starting with patch ' num2str(patch_ind) '/' num2str(length(patch_info_array))]);
    %disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Memory usage:']); whos;
    patch_info = patch_info_array{patch_ind};
    
    %% modify SI structure
    SI_patch = SI;
    SI_patch.outdir = sprintf('%s%spatch_%02d', SI.outdir, filesep(), patch_ind);
    mkdir(SI_patch.outdir);
    
    %% sensor movie loader module
    disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Loading and rectifying input frames']);
    sensor_movie = image_load_module_aug(patch_info, SI_patch);
    disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Done loading and rectifying input frames']);
    save(fullfile(SI_patch.outdir, [datestr(now, 'YYmmddTHHMM') '_patch_info.mat']), 'patch_info', '-v7.3');
    %disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Memory usage:']); whos;
    
    shortcut = false;
    if ~shortcut
        %% registration
        if SI.reg_flag
            disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Starting registration (motion correction)']);
            sensor_movie = LFM_registration_module(sensor_movie, patch_info, SI_patch);
            %disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Memory usage:']); whos;
        end
        
        %% summary image generation module
        [summary_image, bg_spatial_init, bg_temporal_init] = ...
            summary_image_generation_module(sensor_movie, patch_info, SI_patch);
        
        %% reconstruction module
        %     if numel(SI_patch.gpu_ids) > 0
        %         gpu = gpuDevice(SI_patch.gpu_ids(randi(numel(SI_patch.gpu_ids))));
        %     end
        disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Starting reconstruction']);
        recon_stack = LFM_reconstruction_module(summary_image, ext_PSF_struct.H, ext_PSF_struct.Ht, SI_patch);
        
        % remove top and bottom slabs
        recon_stack =  recon_stack(:, :, ceil(size(ext_PSF_struct.H, 5) / 2) - floor(SI_patch.valid_recon_range / 2) :...
            ceil(size(ext_PSF_struct.H, 5) / 2) + floor(SI_patch.valid_recon_range / 2) );
        
        % Full PSF z range no longer needed, so only keep the central
        % SI.valid_recon_range slices
        ext_PSF_struct.H = ext_PSF_struct.H(:, :, :, :, ...
            ceil(size(ext_PSF_struct.H, 5) / 2) - floor(SI.valid_recon_range / 2) :...
            ceil(size(ext_PSF_struct.H, 5) / 2) + floor(SI.valid_recon_range / 2));
        ext_PSF_struct.Ht = ext_PSF_struct.Ht(:, :, :, :, ...
            ceil(size(ext_PSF_struct.H, 5) / 2) - floor(SI.valid_recon_range / 2) : ...
            ceil(size(ext_PSF_struct.H, 5) / 2) + floor(SI.valid_recon_range / 2));
        %disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Memory usage:']); whos;
        
        % checkpoint
        save(fullfile(SI_patch.outdir, [datestr(now, 'YYmmddTHHMM') '_recon.mat']), 'recon_stack', 'summary_image',...
            'bg_spatial_init', 'bg_temporal_init', 'SI_patch', '-v7.3');
        
        %% segmentation module
        disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Starting segmentation']);
        [valid_seg, ~] = neuron_segmentation_module_alter(sensor_movie, recon_stack, SI_patch, patch_info, mask, ext_PSF_struct);
        
        %% library generation module
        disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Starting library generation']);
        [S_init, T_init, S_bg_init, T_bg_init, S_mask_init, S_bg_mask_init, bias] = ...
            seed_library_generation_module(sensor_movie, recon_stack, valid_seg, ext_PSF_struct, ...
            bg_spatial_init, bg_temporal_init, SI_patch);
        
        H_sz = size(ext_PSF_struct.H);
        if patch_ind == my_patch_ixs(end)
            % PSF and sensor_movie no longer needed; save memory
            ext_PSF_struct.H = [];
            ext_PSF_struct.Ht = [];
        end
        
        % Save checkpoint
        save(fullfile(SI_patch.outdir, [datestr(now, 'YYmmddTHHMM') '_initial_components.mat']), 'S_init', 'T_init', 'S_mask_init',...
            'bg_spatial_init', 'bg_temporal_init', 'summary_image', 'valid_seg', 'bias', 'S_bg_init', 'T_bg_init', 'S_bg_mask_init', 'SI_patch', '-v7.3');
        %disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Memory usage:']); whos;
    else
        fns = dir(fullfile(SI_patch.outdir, '*_initial_components.mat'));
        load(fullfile(fns(end).folder, fns(end).name)); %#ok<LOAD>
    end
    
    %% main demixing module
    disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Starting main demixing']);
    [S, T_raw, S_bg, T_bg, bg_spatial, bg_temporal] = ...
        main_demixing_module(sensor_movie, S_init, T_init, ...
        S_bg_init, T_bg_init, S_mask_init, S_bg_mask_init,...
        bg_spatial_init,  bg_temporal_init, bias,  patch_info, SI_patch);
    clear sensor_movie;
    % checkpoint
    save(fullfile(SI_patch.outdir, [datestr(now, 'YYmmddTHHMM') '_final_components.mat']),...
        'S', 'S_bg', 'T_raw', 'T_bg', 'bg_spatial', 'bg_temporal', 'bias', 'SI_patch', '-v7.3');
    %disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Memory usage:']); whos;
    
    %% component refinement
    disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Starting component refinement']);
    [neuron_trace_mat, deconvol_neuron_trace_mat, ...
        spike_mat, ~] = component_refinement_module(T_raw, T_bg, SI_patch);
    % checkpoint
    save(fullfile(SI_patch.outdir, [datestr(now, 'YYmmddTHHMM') '_final_temporal_result.mat']),...
        'neuron_trace_mat', 'deconvol_neuron_trace_mat', 'spike_mat','SI_patch', '-v7.3');
    %disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Memory usage:']); whos;
    
    %% component merging
    [merged_S, merged_bias, merged_T, merged_T_deconv, merged_spike, merged_seg] =...
        component_merge(S, bias, neuron_trace_mat,  deconvol_neuron_trace_mat, spike_mat, valid_seg, ...
        patch_info.size, SI_patch);
    
    keep_ind = true(1, length(merged_S));
    % throw away components that are too large
    for i = 1 : length(merged_S)
        buf = merged_S{i};
        area_size_x = size(buf, 1);
        area_size_y = size(buf, 2);
        % shape should not be too large
        if area_size_x >  (H_sz(1) * 1.5 + round(SI_patch.neuron_lateral_size * 10)) || ...
                area_size_y > (H_sz(2) * 1.5 + round(SI_patch.neuron_lateral_size * 10))
            keep_ind(i) = false;
        end
    end
    merged_S = merged_S(keep_ind);
    merged_T = merged_T(keep_ind, :);
    merged_T_deconv = merged_T_deconv(keep_ind, :);
    merged_bias = merged_bias(keep_ind);
    merged_spike = merged_spike(keep_ind, :);
    merged_seg = merged_seg(keep_ind, :);
    
    % checkpoint
    save(fullfile(SI_patch.outdir, [datestr(now, 'YYmmddTHHMM') '_after_merge.mat']), ...
        'merged_S', 'merged_bias', 'merged_T','merged_T_deconv', 'merged_spike', 'merged_seg', ...
        'bg_spatial', 'bg_temporal', '-v7.3');
    %disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Memory usage:']); whos;
    
end

if SI.worker_ix == 0 || SI.worker_ix > length(patch_info_array)
    %% main demixing test
    %     T_mat = [];
    %     for i = 1 : length(T_raw)
    %         T_mat(i, :) = reshape( T_raw{i}, 1, []);
    %     end
    %     figure, imagesc(T_mat), title('temporal activity');
    %     if ~isempty(T_bg_init)
    %         T_bg_mat = [];
    %         for i = 1 : length(T_bg)
    %             T_bg_mat(i, :) = reshape(T_bg{i}, 1, []);
    %         end
    %         figure, imagesc(T_bg_mat), title('bg')
    %     end
    
    %% Merge patches (patch_merging_module() loads result files for all patches)
    disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Starting patch merging']);
    [global_S, global_bias, global_bg_S, global_bg_T, global_T, global_spike, ...
        global_T_deconv, global_seg] = patch_merging_module(patch_info_array, SI);
    
    save(fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_merged_result.mat']),...
        'global_S', 'global_bias', 'global_bg_S', 'global_bg_T', ...
        'global_T', 'global_spike', 'global_T_deconv', 'global_seg', 'patch_info_array', 'SI', '-v7.3');
    %disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Memory usage:']); whos;
    
    %% quality check
    disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Starting trace quality classification']);
    positive_ind = temporal_activity_filtering(global_T, SI.filt_method, SI.trace_keep_mode, SI.frames_step, SI.gpu_ids(1));
    global_T_filtered = global_T(positive_ind, :);
    global_T_deconv_filtered  = global_T_deconv(positive_ind, :);
    global_spike_filtered  = global_spike(positive_ind, :);
    global_S_filtered = global_S(positive_ind);
    global_seg_filtered = global_seg(positive_ind,:);
    
    x_pixel_size = abs(ext_PSF_struct.x1objspace(2) - ext_PSF_struct.x1objspace(1)) * 1e6;
    z_pixel_size = abs(ext_PSF_struct.x3objspace(2) - ext_PSF_struct.x3objspace(1)) * 1e6;
    global_neuron_positions = zeros(size(global_S,2), 3);
    for i = 1 : size(global_S, 2)  % 1 : length(global_seg)
        pos = global_seg{i, 2};
        pos_mean = mean(pos, 1);
        pos_mean(1) = pos_mean(1) * x_pixel_size;
        pos_mean(2) = pos_mean(2) * x_pixel_size;
        pos_mean(3) = pos_mean(3) * z_pixel_size;
        global_neuron_positions(i, :) = pos_mean;
    end
    global_neuron_positions_filtered = global_neuron_positions(positive_ind, :);
    
    save(fullfile(SI.outdir, 'after_filter_final_temporal_result.mat'),...
        'global_T_filtered', 'global_T_deconv_filtered', 'global_spike_filtered', 'positive_ind', 'global_S_filtered', 'global_seg_filtered', 'global_neuron_positions', 'global_neuron_positions_filtered', '-v7.3');
    %disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Memory usage:']); whos;
    
    %% Plot stacked subset of temporal activity traces, two columns
%     if SI.summary_plots_enable
%         disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Plotting results']);
%         ts = zscore(global_T_filtered, 0, 2);
%         y_shift = 4;
%         clip = true;
%         rng(10021)
%         if size(ts,1) > 100
%             sel = randperm(size(ts,1), 100);
%         else
%             sel = 1:size(ts,1);
%         end
%         nixs = 1:size(ts,1);
%         sel_nixs = nixs(sel);
% 
%         figure('Position', [10 10 2000 2000]);
%         title('Raw traces, z-scored, after final classifier', 'Interpreter', 'none');
%         subplot(121);
%         hold on
%         for n_ix = 1:floor(numel(sel_nixs)/2)
%             ax = gca();
%             ax.ColorOrderIndex = 1;
%             loop_ts = ts(sel_nixs(n_ix),:);
%             if clip
%                 loop_ts(loop_ts > 3*y_shift) = y_shift;
%                 loop_ts(loop_ts < -3*y_shift) = -y_shift;
%             end
%             t = (0:size(ts,2)-1);
%             plot(t, squeeze(loop_ts) + y_shift*(n_ix-1));
%             text(30, y_shift*(n_ix-1), num2str(sel_nixs(n_ix)));
%         end
%         xlabel('Frame');
%         xlim([min(t) max(t)]);
%         hold off;
%         axis tight;
%         set(gca,'LooseInset',get(gca,'TightInset'))
% 
%         subplot(122);
%         hold on;
%         for n_ix = ceil(numel(sel_nixs)/2):numel(sel_nixs)
%             ax = gca();
%             ax.ColorOrderIndex = 1;
%             loop_ts = ts(sel_nixs(n_ix),:);
%             if clip
%                 loop_ts(loop_ts > y_shift) = y_shift;
%                 loop_ts(loop_ts < -y_shift) = -y_shift;
%             end
%             t = (0:size(ts,2)-1);
%             plot(t, squeeze(loop_ts) + y_shift*(n_ix-1));
%             text(30, y_shift*(n_ix-1), num2str(sel_nixs(n_ix)));
%         end
%         xlabel('Frame');
%         xlim([min(t) max(t)]);
%         hold off;
%         axis tight;
%         set(gca,'LooseInset',get(gca,'TightInset'))
%         saveas(gca, fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_stacked_temporal_components.png']));
%         close(gcf);
%     end
    
    %% Plot stacked raw signals
    if SI.summary_plots_enable 
        plotControl.frameRate = SI.frameRate;
        plotControl.normalization = 1;
        plotControl.sep = 0.8; % vertical separation
        plotControl.displayLabel = 0;
        plotControl.plotInferred = 1; % blue
        plotControl.plotFiltered = 0; % pink
        plotControl.plotRaw = 1; % grey
        plotControl.rollingView = 0;
        plotControl.maxNnum = 1000;
        %plotControl.maxNnum = 300;
        plotActivityTrace(global_T_filtered, global_T_filtered,...
            global_T_filtered, plotControl) % true_gt
        set(gca,'YTickLabel',[]);
        set(gca,'YLabel',[]);
        title('Raw traces, after final classifier')
        box on
        ax = gca;
        outerpos = ax.OuterPosition;
        ti = ax.TightInset;
        left = outerpos(1) + ti(1);
        bottom = outerpos(2) + ti(2);
        ax_width = outerpos(3) - ti(1) - ti(3);
        ax_height = outerpos(4) - (ti(2) - ti(4)) * 1.8;
        ax.Position = [left bottom ax_width ax_height];
        t = title(SI.outdir);
        t.Interpreter = 'None';
        %pbaspect([1 1 1])
        saveas(gca, fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_temporal_components.png']));
        savefig(fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_temporal_components.fig']));
        close(gcf);
    end
    
    %% Plot neuron 3D positions, with maximum projection
    if SI.summary_plots_enable
        disp('Plot neuron 3D positions')
        plot_pos = global_neuron_positions;
        FOV_x = max(plot_pos(:, 1)) - min(plot_pos(:, 1));
        FOV_y = max(plot_pos(:, 2)) - min(plot_pos(:, 2));
        FOV_z = max(plot_pos(:, 3)) - min(plot_pos(:, 3));

        figure;
        scatter3(plot_pos(:, 1), ...
            plot_pos(:, 2),...
            plot_pos(:, 3), 10, 'filled');
        ax = gca;
        ax.BoxStyle = 'full';
        box on
        axis equal
        axis vis3d % important, not change the size
        xlabel('x axis [um]')
        ylabel('y axis [um]')
        zlabel('z axis [um] (greater=deeper)')

        xticks([1, ceil(FOV_x /2), FOV_x])
        xticklabels({'0', ...
            sprintf('%.0f', FOV_x / 2),...
            sprintf('%.0f', FOV_x)})
        yticks([1, ceil(FOV_y /2), FOV_y])
        yticklabels({'0', ...
            sprintf('%.0f', FOV_y / 2 ),...
            sprintf('%.0f', FOV_y)})
        zticks([1, ceil(FOV_z /2), FOV_z])
        zticklabels({'0', ...
            sprintf('%.0f', FOV_z / 2 ),...
            sprintf('%.0f', FOV_z)})

        view(60, 20)
        set(gca,'color','none')
        saveas(gca, fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_spatial_position_volume.png']));
        savefig(fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_spatial_position_volume.fig']));

        set(gca, 'XTickLabel', []);set(gca, 'YTickLabel', []);set(gca, 'ZTickLabel', []);
        view(90, 90)
        set(gca,'color','none')
        saveas(gca, fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_spatial_position_top.png']));

        view(90, 0)
        set(gca,'color','none')
        saveas(gca, fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_spatial_position_front.png']));

        view(180, 0)
        set(gca,'color','none')
        saveas(gca, fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_spatial_position_right.png']));
    end
    
%     %% Plot neuron histogram vs z
%     figure;
%     histogram(global_neuron_positions_filtered(:, 3), 21);
%     xlabel('z [um]');
%     ylabel('Frequency');
%     saveas(gca, fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_pos_hist_z.png']));
%     savefig(fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_pos_hist_z.fig']));
%     ylabel('First image array index, increasing left to right');
    
    %% Plot quality metric
    if SI.summary_plots_enable
        ts1 = global_T_filtered;
        [~,~,~,~,explained1,~] = pca(ts1);
        ix1_95 = find(cumsum(explained1) > 95, 1, 'first');
        figure;
        hold on;
        plot(explained1, 'b+-');
        xline(ix1_95, 'b', {['95% variance explained: ' num2str(ix1_95) ' components']});
        hold off;
        ylabel('Percent explained variance');
        xlabel('PCA component no.');
        title(SI.outdir, 'Interpreter', 'none');
        saveas(gca, fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_pca_quality.png']));
    end
    
    %% Print summary statistics
    fn = fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_summary.txt']);
    fid = fopen(fn, 'w');
    fprintf(fid, 'Total no. of neurons pre filtering:\t\t%d \n', size(global_T, 1));
    fprintf(fid, 'Total no. of neurons post filtering:\t\t%d \n', size(global_T_filtered, 1));
    fprintf(fid, 'No. of PCA components required to explain 95 precent of variance:\t\t%d \n', ix1_95);
    fprintf(fid, '[xmin xmax dx] (mm, pre filter):\t\t%4.3f\t%4.3f\t%4.3f\n', min(global_neuron_positions_filtered(:,1))/1000, max(global_neuron_positions_filtered(:,1)/1000), max(global_neuron_positions_filtered(:,1)/1000) - min(global_neuron_positions_filtered(:,1))/1000);
    fprintf(fid, '[ymin ymax dz] (mm, pre filter):\t\t%4.3f\t%4.3f\t%4.3f\n', min(global_neuron_positions_filtered(:,2))/1000, max(global_neuron_positions_filtered(:,2)/1000), max(global_neuron_positions_filtered(:,2)/1000) - min(global_neuron_positions_filtered(:,2))/1000);
    fprintf(fid, '[zmin zmax] (um, pre filter):\t\t%6.1f\t%6.1f\n', min(global_neuron_positions_filtered(:,3)), max(global_neuron_positions_filtered(:,3)));
    fclose(fid);
    type(fn);
end

%% Clean up
gpuDevice([]);
delete(gcp('nocreate'));
rmdir(job_storage_location, 's');

exit_code = 0;
disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ' Returning with exit code ' num2str(exit_code)]);
