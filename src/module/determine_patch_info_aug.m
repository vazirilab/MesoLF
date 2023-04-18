function patch_info_array = determine_patch_info_aug(SI)
%% Determine patch sizes/locations from user input in SI struct, both in raw and rectified frames
%  two modes can be chosen: one is 'carpet', which will tile
%  the entire input image with pathes. The other is 'ROI', which allows to manually
%  specify the ROI locations and patch size (each ROI will be tiled with
%  patches of specified size).
%  The returned array of patch_info structs contains the following fields for each patch:
%       - "size" and "location" of patch in rectified image
%       - "raw_cut":  vector; patch location in raw image for specified region read from tif file
%       - "raw_cut_xCenter", "raw_cut_yCenter": center of central microlens in raw_cut patch, for rectification

%% Parse relevant params in SI struct
xCenter = SI.frames_x_offset;
yCenter = SI.frames_y_offset;
dx = SI.frames_dx;
Crop = true;
XcutLeft = SI.frames_crop_border_microlenses(1);
XcutRight = SI.frames_crop_border_microlenses(2);
YcutUp = SI.frames_crop_border_microlenses(3);
YcutDown= SI.frames_crop_border_microlenses(4);
img_path = SI.indir;
M = SI.Nnum;

patch_mode = SI.patch_mode;
if strcmp(patch_mode, 'carpet')
    n_patches = SI.n_patches;
    % if the input n_patches is a scalar
    if length(n_patches) == 1
        n_patches = [n_patches, n_patches];
    end
elseif strcmp(patch_mode, 'roi')
    ROI_list= SI.ROI_list;
    try
        ROI_patch_size = SI.ROI_patch_size;
        if mod(ROI_patch_size, M)
            warning('SI.ROI_patch_size should be an integer multiple of SI.Nnum. Reducing it to meet this condition...')
            % fix patch size
            ROI_patch_size = ROI_patch_size - mod(ROI_patch_size, M);
        end
    catch
        error('In ROI mode, SI.ROI_patch_size is required')
    end
end

%% Read, rectify and crop one full frame to serve as a reference for further steps below
dinfo = dir(sprintf('%s%s*.tif*', img_path, filesep));
thisfilename = dinfo(1).name;  %just the name
IMG_BW = double(loadtiff(sprintf('%s%s%s', img_path, filesep, thisfilename))); %load just this file

Xresample = [fliplr((xCenter):-dx/M:1)  ((xCenter)+dx/M:dx/M:size(IMG_BW,2))]; % why plus one?
Yresample = [fliplr((yCenter):-dx/M:1)  ((yCenter)+dx/M:dx/M:size(IMG_BW,1))];

% old grid, integral
x0 = 1 : size(IMG_BW, 2);
y0 = 1 : size(IMG_BW, 1);

Mdiff = floor(M/2);

% make sure integral number of microlens, cut left
XqCenterInit = find(Xresample == (xCenter)) - Mdiff; % left
XqInit = XqCenterInit -  M*floor(XqCenterInit/M);

YqCenterInit = find(Yresample == (yCenter)) - Mdiff;
YqInit = YqCenterInit -  M*floor(YqCenterInit/M);

XresampleQ = Xresample(XqInit + 1:end); % what if XqInit is zero?
YresampleQ = Yresample(YqInit+ 1:end);

pre_centerX = find(Xresample == (xCenter)); % overall center ind in Xresample axis
pre_centerY = find(Yresample == (yCenter)); % overall center ind in Xresample axis

% griddedInterpolant doesn't support GPU
% if isa(gpu_device, 'parallel.gpu.CUDADevice')
%     IMG_BW = gpuArray(IMG_BW);
% end
GI = griddedInterpolant({y0, x0}, IMG_BW);
IMG_RESAMPLE = gather(GI({YresampleQ, XresampleQ}));

% cut right
IMG_RESAMPLE = IMG_RESAMPLE((1 : 1 : M*floor((size(IMG_RESAMPLE,1))/M)), ...
    (1 : 1 : M*floor((size(IMG_RESAMPLE,2))/M)));

% Crop
if Crop
    XsizeML = size(IMG_RESAMPLE,2)/M;
    YsizeML = size(IMG_RESAMPLE,1)/M;
    if (XcutLeft + XcutRight)>=XsizeML
        error('X-cut range is larger than the x-size of image');
    end
    if (YcutUp + YcutDown)>=YsizeML
        error('Y-cut range is larger than the y-size of image');
    end
    
    Xrange = (1+XcutLeft:XsizeML-XcutRight);
    Yrange = (1+YcutUp:YsizeML-YcutDown);
    
    IMG_RESAMPLE_crop2 = IMG_RESAMPLE(   ((Yrange(1)-1)*M+1 :Yrange(end)*M),  ((Xrange(1)-1)*M+1 :Xrange(end)*M) );
else
    IMG_RESAMPLE_crop2 = IMG_RESAMPLE;
end

%% Determine size and number of patches to generate
if strcmp(patch_mode, 'carpet')
    
    % intensively cover all the image area
    [global_size_h, global_size_w] = size(IMG_RESAMPLE_crop2);
    patch_size_h = floor(global_size_h / n_patches(1));
    patch_size_w = floor(global_size_w / n_patches(2));
    
    % crop the patch size to fit with M
    patch_size_h = ceil(patch_size_h / M) * M;
    patch_size_w = ceil(patch_size_w / M)* M;
    % determine the start and end corner
    patch_info_array = [];
    for i = 1 : n_patches(1) * n_patches(2)
        [curr_id_i, curr_id_j] = ind2sub([n_patches(1), n_patches(2)], i);
        top_left = [(curr_id_i - 1) * patch_size_h + 1, (curr_id_j - 1) * patch_size_w + 1];
        
        % patch location
        bottom_right = [min(curr_id_i * patch_size_h, global_size_h), ...
            min(curr_id_j * patch_size_w, global_size_w)];
        
        % udpate patch size
        patch_size_h_new = bottom_right(1) - top_left(1) + 1;
        patch_size_w_new = bottom_right(2) - top_left(2) + 1;
        
        patch_size_h_new = patch_size_h_new - mod(patch_size_h_new, M);
        patch_size_w_new = patch_size_w_new - mod(patch_size_w_new, M);
        
        if patch_size_h_new < 50 || patch_size_w_new < 50
            warning('No. %d patch in is too small, discard', i)
            continue
        end
        bottom_right(1)  = top_left(1) + patch_size_h_new - 1;
        bottom_right(2)  = top_left(2) + patch_size_w_new - 1;
        patch_loc = [top_left; bottom_right];
        patch_info.location = patch_loc;
        patch_info.size = [patch_size_h_new, patch_size_w_new];
        
        patch_info_array_length = length(patch_info_array);
        patch_info_array{patch_info_array_length + 1} = patch_info;
    end
    % options for ROI mode
elseif strcmp(patch_mode, 'roi')
    patch_info_array = [];
    % parser
    
    %     if ~iscell(ROI_list)
    %         error('in ROI mode, a cell array contains positions of each ROI is required')
    %     end
    
    ROI_num = size(ROI_list,1);
    % for different ROI
    for i = 1 : ROI_num
        % read curr ROI
        curr_ROI = ROI_list(i,:);
        
        % read corner information
        curr_ROI_top_left = curr_ROI(1:2);
        curr_ROI_bottom_right = curr_ROI(3:4);
        
        % determine the size of current ROI
        curr_ROI_size = [curr_ROI_bottom_right(1) - curr_ROI_top_left(1) + 1, ...
            curr_ROI_bottom_right(2) - curr_ROI_top_left(2) + 1];
        
        % determine patch number
        patch_num_h = ceil(curr_ROI_size(1) / ROI_patch_size);
        patch_num_w = ceil(curr_ROI_size(2) / ROI_patch_size);
        
        % safe check for the patch size
        if curr_ROI_size(1) <  ROI_patch_size
            warning('No. %d ROI size is smaller than patch size in height \n', i)
            ROI_patch_size_new_h = curr_ROI_size(1);
            patch_num_h = 1;
        else
            ROI_patch_size_new_h = ROI_patch_size;
        end
        if curr_ROI_size(2) <  ROI_patch_size
            warning('No. %d ROI size is smaller than patch size in width \n', i)
            ROI_patch_size_new_w = curr_ROI_size(2);
            patch_num_w = 1;
        else
            ROI_patch_size_new_w = ROI_patch_size;
        end
        
        % for all patches in i-th roi
        for j = 1 :patch_num_h * patch_num_w
            [curr_id_i, curr_id_j] = ind2sub([patch_num_h , patch_num_w], j);
            top_left = [(curr_id_i - 1) * ROI_patch_size_new_h + 1, ...
                (curr_id_j - 1) * ROI_patch_size_new_w + 1] + curr_ROI_top_left - 1;
            
            % patch location
            bottom_right = [min(curr_id_i * ROI_patch_size_new_h, curr_ROI_size(1)), ...
                min(curr_id_j * ROI_patch_size_new_w, curr_ROI_size(2))] + curr_ROI_top_left - 1;
            
            % udpate patch size
            patch_size_h = bottom_right(1) - top_left(1) + 1;
            patch_size_w = bottom_right(2) - top_left(2) + 1;
            
            patch_size_h = patch_size_h - mod(patch_size_h, M);
            patch_size_w = patch_size_w - mod(patch_size_w, M);
            if patch_size_h < 50 || patch_size_w < 50
                warning('No. %d patch in No. %d ROI is too small (less than 50 px wide or tall), discarding.', j, i)
                continue
            end
            bottom_right(1)  = top_left(1) + patch_size_h - 1;
            bottom_right(2)  = top_left(2) + patch_size_w - 1;
            
            patch_loc = [top_left; bottom_right];
            patch_info.location = patch_loc;
            patch_info.size = [patch_size_h, patch_size_w];
            
            patch_info_array_length = length(patch_info_array);
            patch_info_array{patch_info_array_length + 1} = patch_info;
        end
    end
    
else
    error('patch_mode should be either ''carpet'' or ''roi''');
end

%% Determine center and size of desired post-rectification patch in raw image (required for partial load of raw frame)
% coordinate after rectification, all integers
Y_after_top = (Yrange(1)-1)*M+1; % crop, first coord align due to a global crop
X_after_left = (Xrange(1)-1)*M+1;

for i = 1 : length(patch_info_array)
    % grab patch info that after rectification
    patch_info = patch_info_array{i};
    top_after = patch_info.location(1, 1);
    left_after =  patch_info.location(1, 2);
    bottom_after =  patch_info.location(2, 1);
    right_after =  patch_info.location(2, 2);
    
    margin = ceil(M / 2) - 2;
    % apply main rectification shift, which determine the shape and center
    patch_top_in_global_after = top_after + ...
        Y_after_top - 1 ... % manual crop
        + YqInit - 1 ... % automatical crop
        - margin; % safe margin
    patch_left_in_global_after = left_after + ...
        X_after_left - 1 ...
        + XqInit - 1 ...
        - margin;
    patch_bottom_in_global_after = bottom_after + ...
        Y_after_top - 1 ...
        + YqInit - 1 ...
        + margin - 1;
    patch_right_in_global_after = right_after + ...
        X_after_left - 1 ...
        + XqInit - 1 ...
        + margin - 1;
    
    % trace back to the original coordinate, make sure the patch to be
    % larger
    Yorigin_start = max(floor(Yresample(patch_top_in_global_after)), 1);
    Yorigin_end = min(ceil(Yresample(patch_bottom_in_global_after)), size(IMG_BW, 1));
    Xorigin_start = max(floor(Xresample(patch_left_in_global_after)), 1);
    Xorigin_end = min(ceil(Xresample(patch_right_in_global_after)), size(IMG_BW, 2));
    
    % get patch center after mapping
    patch_center_y = round((patch_top_in_global_after + patch_bottom_in_global_after) / 2);
    patch_center_x = round((patch_left_in_global_after + patch_right_in_global_after) / 2);
    
    % patch center, such that it is the center of microlens, in global, in
    % original coordinate
    if mod(pre_centerY - patch_center_y, M) >= ceil(M / 2)
        yCenter_patch = Yresample(patch_center_y - (M - mod(pre_centerY - patch_center_y, M))); % incoorperate the M
    else
        yCenter_patch = Yresample(patch_center_y + mod(pre_centerY - patch_center_y, M)); % incoorperate the M
    end
    
    
    if  mod(pre_centerX - patch_center_x, M) >= ceil(M /2)
        xCenter_patch = Xresample(patch_center_x - (M - mod(pre_centerX - patch_center_x, M)));
    else
        xCenter_patch = Xresample(patch_center_x + mod(pre_centerX - patch_center_x, M));
    end
    % subtract the bias
    yCenter_patch = yCenter_patch - Yresample(patch_top_in_global_after);
    xCenter_patch = xCenter_patch - Xresample(patch_left_in_global_after);
    
    patch_info_array{i}.raw_cut = [Yorigin_start, Yorigin_end; Xorigin_start, Xorigin_end];
    patch_info_array{i}.raw_cut_xCenter = xCenter_patch;
    patch_info_array{i}.raw_cut_yCenter = yCenter_patch;
end

figure;
imagesc(IMG_RESAMPLE_crop2);
axis image;
colorbar;
title(['Full frame after rectification and cropping. Size: ' num2str(size(IMG_RESAMPLE_crop2))]);
for i = 1 : length(patch_info_array)
    p = patch_info_array{i};
    x = p.location(1,1);  %first row is top left, second row is bottom right
    y = p.location(1,2);
    w = p.size(1);
    h = p.size(2);
    rectangle('Position', [y x h w]);  % extremely confusingly, the coordinates that rectangle() uses are flipped w.r.t. the way imagesc uses and labels the axes
    text(y + 40, x + 40, num2str(i));
end
xlabel('Second image array index, increasing left to right');
ylabel('First image array index, increasing top to bottom');
savefig(fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_frame_rois_post_rect_crop.fig']));

end
