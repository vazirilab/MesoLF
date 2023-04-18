function sensor_movie = image_load_module_aug(patch_info, SI)
%% module for loading sensory image of LFM.
%   Input
%   patch_info: structure, contains patch location, raw  and patch size
%   SI: configuration structure
%       SI.frames.x_offset and SI.frames.y_offset: center of one of
%       microlens
%       SI.dx: size of microlens
%       SI.crop_border_microlens: discarded boundaries
%       SI.indir: path that stores images
%   Output
%   sensor_movie: read movie of current patch


%   last update: 7/7/2020. YZ

%% parser
% xCenter = SI.frames_x_offset;
% yCenter = SI.frames_y_offset;
dx = SI.frames_dx;
% Crop = true;
% if numel(SI.gpu_ids) > 0
%     gpu_ix = mod(SI.worker_ix, numel(SI.gpu_ids)) + 1;
%     gpu_device = gpuDevice(SI.gpu_ids(gpu_ix));
% else
%     gpu_device = false;
% end

% XcutLeft = SI.frames_crop_border_microlenses(1);
% XcutRight= SI.frames_crop_border_microlenses(3);
% YcutUp= SI.frames_crop_border_microlenses(3);
% YcutDown= SI.frames_crop_border_microlenses(4);
img_path = SI.indir;
M = SI.Nnum;

% temporal parameter
frame_start = SI.frames_start;
frame_end = SI.frames_end;
frame_step = SI.frames_step;

dinfo = dir(sprintf('%s%s*.tif*', img_path, filesep));

if SI.frames_average
    frames_n_average = frame_step;
    indx_array = frame_start : 1 : min(length(dinfo), frame_end);
else
    frames_n_average = 1;
    indx_array = frame_start : frame_step : min(length(dinfo), frame_end);
end

frame_n_final = length(frame_start : frame_step : min(length(dinfo), frame_end));

patch_size_h = patch_info.size(1);
patch_size_w = patch_info.size(2);
raw_cut_start = patch_info.raw_cut;
raw_cut_xCenter = patch_info.raw_cut_xCenter;
raw_cut_yCenter = patch_info.raw_cut_yCenter;

sensor_movie = ones(patch_size_h * patch_size_w, frame_n_final, 'single');

%% load progress
disp('Reading frames: ');
%cluster = parcluster();
% n_workers_sav = cluster.NumWorkers;
% cluster.NumWorkers = n_workers;
% cluster.NumThreads = 4;
% delete(gcp('nocreate'));
% parpool(n_workers);
for i = 1 : frame_n_final
    for j = 0 : (frames_n_average - 1)
        %textprogressbar(i / frameN * 100);
        frame_ix = (i - 1) * frames_n_average + j + 1;
        avg_rng_end = j + 1;
        if frame_ix > length(indx_array)
            break
        end
        thisfilename = dinfo(indx_array(frame_ix)).name;
        
        if mod(i, 25) == 1
            fprintf('%i/%i (%s) ', i, frame_n_final, thisfilename);
        end
        if mod(i, 125) == 0
            fprintf('\n');
        end
        
        % load small area
        curr_patch = imread(sprintf('%s%s%s', img_path, filesep, thisfilename),...
            'PixelRegion', {raw_cut_start(1, :), raw_cut_start(2, :)});  % The first vector specifies the range of rows to read, and the second vector specifies the range of columns to read.
        curr_patch = double(curr_patch);
        if j == 0
            patches_to_average = zeros([frames_n_average size(curr_patch)]);
        end
        patches_to_average(j + 1, :, :) = curr_patch;
    end
    curr_patch = squeeze(mean(patches_to_average(1:avg_rng_end, :, :), 1));
    
    % apply rectify for small patches
    IMG_RESAMPLE = img_rectify_mod(curr_patch, raw_cut_xCenter, ...
        raw_cut_yCenter, M, dx);
    
    new_patch_size_h = size(IMG_RESAMPLE, 1);
    new_patch_size_w = size(IMG_RESAMPLE, 2);
    
    if patch_size_h ~= new_patch_size_h || patch_size_w ~= new_patch_size_w
        error('After raw patch read and rectification, the shape is different from what was expected.')
    end

    sensor_movie(:, i) = IMG_RESAMPLE(:);
end
fprintf('\n ...done.\n');

%%
% figure;
% imagesc(squeeze(sensor_movie(:, :, 1)));
% axis image;
% colorbar;
% print(gcf, fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_patch_first_frame.png']), '-dpng', '-r300');
% savefig(gcf, fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_patch_first_frame.fig']));

%sensor_movie = reshape(sensor_movie, [], frameN);
% normalization is necessary
sensor_movie_max = max(sensor_movie(:));
sensor_movie = sensor_movie ./ sensor_movie_max;
end

%% utility function
function IMG_RESAMPLE = img_rectify_mod(IMG_BW, xCenter, yCenter, M, dx)
% note this function is different ImageRect
% 1. we querry the position of xCenter, instead of xCenter + 1
% 2. the left part margin is 0.5 instead of 1, such that the image will be
% slightly bigger
Xresample = [fliplr((xCenter):-dx/M: -dx/M)  ((xCenter)+dx/M:dx/M:size(IMG_BW,2) + 0.9)];
Yresample = [fliplr((yCenter):-dx/M: -dx/M)  ((yCenter)+dx/M:dx/M:size(IMG_BW,1) + 0.9)];

% old grid, integral
x0 = 1 : size(IMG_BW, 2);
y0 = 1 : size(IMG_BW, 1);

Mdiff = floor(M/2);

% make sure integral number of microlens, cut left
XqCenterInit = find(Xresample == (xCenter)) - Mdiff; % left
XqInit = XqCenterInit -  M*floor(XqCenterInit/M);



YqCenterInit = find(Yresample == (yCenter)) - Mdiff;
YqInit = YqCenterInit -  M*floor(YqCenterInit/M);

XresampleQ = Xresample(XqInit + 1:end);
YresampleQ = Yresample(YqInit + 1:end);

% griddedInterpolant doesn't support GPU
% if isa(gpu_device, 'parallel.gpu.CUDADevice')
%     IMG_BW = gpuArray(IMG_BW);
% end
GI = griddedInterpolant({y0, x0}, IMG_BW);

IMG_RESAMPLE = gather(GI({YresampleQ, XresampleQ}));

% cut right
IMG_RESAMPLE = IMG_RESAMPLE((1 : 1 : M*floor((size(IMG_RESAMPLE,1))/M)), ...
    (1 : 1 : M*floor((size(IMG_RESAMPLE,2))/M)));
end
