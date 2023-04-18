function [summary_image, bg_spatial_init,bg_temporal_init] = ...
    summary_image_generation_module(sensor_movie, patch_info, SI)

%% module for loading sensory image of LFM.
%   Input
%   sensor_movie: xyt data
%   patch_info: size and location info of current patch
%   SI: configuration structure
%       SI.detrend_mode: summary image calculation mode, you can choose
%       global, and pixelwise
%       detailed parameter see the parser section
%   Output
%   summary_image: summarized image (xy data)
%   bg_spatial_init,bg_temporal_init: initilaized rank-1 spatial and
%   temporal background activities

%   last update: 5/31/2020. YZ

%% parser
Nnum = SI.Nnum;
size_h = patch_info.size(1);
size_w = patch_info.size(2);
movie_size = [size_h, size_w];
outdir = SI.outdir;
buf = size(sensor_movie);
frameN = buf(end);
% parser with different method
detrend_mode = SI.detrend_mode;

if strcmp(detrend_mode, 'global')
    slide_windows_size = SI.global_detrend_delta;
    frame_step = SI.frames_step;
    bg_iter = SI.bg_iter;
elseif strcmp(detrend_mode, 'pixelwise')  
    bg_iter = SI.bg_iter;
    window_1 = SI.pixelwise_window_1;
    window_2 = SI.pixelwise_window_2;
    poly_index = SI.pixelwise_poly_index;
else
    error('Invalid value for param "detrend_mode"');
end
    
%% main body
if strcmp(detrend_mode,'global')
    % get baseline
    baseline_raw = squeeze(mean(sensor_movie,1))';
    smooth_window_span = 2 *  slide_windows_size / frame_step;
    % smooth baseline
    baseline = smooth(baseline_raw, smooth_window_span, 'sgolay', 3);
    figure; hold on; plot(baseline_raw); plot(baseline); title('Frame means (post bg subtract), raw + trend fit'); hold off;
    print(fullfile(outdir, [datestr(now, 'YYmmddTHHMM') '_trend_fit.pdf']), '-dpdf', '-r300');
    
    % devide by the baseline
    sensor_movie_max = max(sensor_movie(:));
    sensor_movie = sensor_movie/sensor_movie_max;
    
    [bg_spatial_init, bg_temporal_init] = rank_1_factorization(sensor_movie,bg_iter);
    summary_image = compute_std_image(sensor_movie, bg_spatial_init, bg_temporal_init); % also input the spatial and temporal background
    
    bg_spatial_init = reshape(bg_spatial_init, movie_size(1:2)); 
    summary_image = reshape(summary_image, movie_size(1:2));
elseif strcmp(detrend_mode, 'pixelwise')
    %disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Memory usage:']); whos;
    % rank 1 background removal
	sensor_movie_max = max(sensor_movie(:));
    [bg_spatial_init, bg_temporal_init] = rank_1_factorization(sensor_movie, bg_iter, sensor_movie_max);
    bg_spatial_init = reshape(bg_spatial_init, movie_size(1:2)); 
    
    % the following line duplicates sensor_movie, but for now we can live with that. Just don't create more copies...
    %sensor_movie = sensor_movie ./ sensor_movie_max; 
    %sensor_movie = reshape(sensor_movie, movie_size(1), movie_size(2), []);
     
    % let's do the following in chunks of 2GB to save memory
    chunk_size = floor((2 * 1024^3 / 8) / size(sensor_movie, 2));
    d1a = size(sensor_movie, 1);
    summary_image = zeros(d1a, 1);
    for i = 1:chunk_size:d1a
        %ACf(i:min(i+chunk_size-1,d1a)) = (A_mat(i:min(i+chunk_size-1,d1a), 1:nr) * C_mat(1:nr, :) + A_mat(i:min(i+chunk_size-1,d1a), nr+2 : end) * C_mat(nr+2 : end, :)) * f';
        sensor_movie_chunk = sensor_movie(i:min(i+chunk_size-1,d1a), :) ./ sensor_movie_max;
        %sensor_movie_chunk = reshape(sensor_movie_chunk, movie_size(1), movie_size(2), []);
        sensor_movie_chunk = sensor_movie_chunk - smoothdata(sensor_movie_chunk, 2, 'movmean', window_1);
%         if i == 1
%             figure, plot(squeeze(sensor_movie_chunk(round(end / 2), round(end / 2), :)));
%         end
        % do a smooth to avoid noise contribute to std
        % window_2 = 20;
        sensor_movie_chunk = smoothdata(sensor_movie_chunk, 2,'movmean', window_2);
%         if 1 == 1
%             hold on, plot(squeeze(sensor_movie_chunk(round(end / 2), round(end / 2), :)));
%             print(fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_example_pixelwise_fit.pdf']), '-dpdf', '-r300');
%         end
        sensor_movie_chunk = sensor_movie_chunk .^ poly_index;
        summary_image(i:min(i+chunk_size-1,d1a)) = compute_std_image(...
            sensor_movie_chunk, ...
            zeros(numel(i:min(i+chunk_size-1,d1a)), 1), ...
            zeros(1, frameN));
    end
%         % detrend
%     %  window_1 = 200; % large window, which is much larger than a calcium transients
%     % unfortunately, this duplicates the input data some more
%     sensor_movie = reshape(sensor_movie, movie_size(1), movie_size(2), frameN);
%     sensor_movie = sensor_movie - smoothdata(sensor_movie, 3, 'movmean', window_1);
%     figure, plot(squeeze(sensor_movie(round(end / 2), round(end / 2), :)));
% 
%     % do a smooth to avoid noise contribute to std
%     % window_2 = 20;    
%     sensor_movie = smoothdata(sensor_movie, 3,'movmean', window_2);
%     hold on, plot(squeeze(sensor_movie(round(end / 2), round(end / 2), :)));
% 	print(fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_example_pixelwise_fit.pdf']), '-dpdf', '-r300');
% 
%     sensor_movie = sensor_movie .^ poly_index;
% 
%     summary_image = compute_std_image(...
%         reshape(sensor_movie, [], frameN), ...
%         zeros(movie_size(1) * movie_size(2), 1), ...
%         zeros(1, frameN));
    
    % Do two smoothdata() steps and the stddev computation in one (hard-to-read) line to
    % avoid duplication of sensor_movie in memory
%     summary_image = compute_std_image(...
%         reshape(...
%             smoothdata(reshape(sensor_movie, movie_size(1), movie_size(2), []) ./ sensor_movie_max - smoothdata(reshape(sensor_movie, movie_size(1), movie_size(2), []) ./ sensor_movie_max, 3, 'movmean', window_1), ...
%                        3, 'movmean', window_2) .^ poly_index, ...
%             [], frameN), ...
%         zeros(movie_size(1) * movie_size(2), 1), ...
%         zeros(1, frameN));
    summary_image = reshape(summary_image,  movie_size(1), movie_size(2));
end
clear sensor_movie;

%% save and return
%disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Memory usage:']); whos;
summary_image = summary_image / max(summary_image(:));
saveastiff(im2uint16(summary_image), ...
    fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_summary_img.tif']));
figure, imagesc(summary_image), axis equal, axis off, title('Summary image without global bkgnd.');
saveas(gca, fullfile(outdir, [datestr(now, 'YYmmddTHHMM') '_summary_image.png']));

figure, imagesc(bg_spatial_init), axis equal, axis off, title('Rank-1 spatial bkgnd.');
saveas(gca, fullfile(outdir, [datestr(now, 'YYmmddTHHMM') '_bg_spatial_rank1.png']));

figure, plot(bg_temporal_init), title('Rank-1 temporal bkgnd.');
saveas(gca, fullfile(outdir, [datestr(now, 'YYmmddTHHMM') '_bg_temporal_rank1.png']));
end
