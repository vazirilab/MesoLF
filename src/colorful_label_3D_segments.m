function colorful_label_3D_segments(segments, ...
        location, size_h, size_w, size_z,std_recon_img, output_dir, name, cut_box)
%% generate a hyper stack to show different components
%  Input:
%       segments: a cell array with N elements, each element (still a cell) 
%           contains many patches to show the segments
%       location: a cell array (same structure as segments) to show the
%           center of each segments
%       size_h, size_w, size_z: size of volume
%       cut_box: cut the final volume
%       output_dir: output directory

%  last update: 2/14/2020. YZ

valid_num = length(segments);
patch_size = size(segments{1}{1}, 1);
img_valid = uint8(zeros(size_h, size_w, 3, size_z));

std_recon_img = double(std_recon_img);
std_recon_img = std_recon_img / max(std_recon_img(:));
std_recon_img = im2uint8(std_recon_img);

valid_in_std_recon_img = std_recon_img;
valid_in_std_recon_img = repmat(valid_in_std_recon_img, [1, 1, 1, 3]);
valid_in_std_recon_img = permute(valid_in_std_recon_img, [1, 2, 4, 3]);
% do a scale for clear background image
colors = rand(valid_num, 3) * 0.5; 
boundary_thickness = 3;
erode_se = strel('square', boundary_thickness * 2);
for i = 1 : valid_num
    disp(i);
	% generate volume
    [volume, ~] = volume_calculator(location{i}, ...
            patch_size, size_h, size_w, size_z);
    for j = 1 : size_z
        
        % add segment label
        slice = squeeze(img_valid(:, :, :, j)); % 3 channel image
        bw_seg = logical(volume(:, :, j));
        slice = labeloverlay(slice, bw_seg, 'colormap', colors(i, :));
    
        % add component number
        ind = find(bw_seg);
        if ~isempty(ind)           
            [sub_j, sub_i] = ind2sub([size_h, size_w], ind(1));
            try  % insertText() is in the Computer Vision Toolbox and might not be available
                slice = insertText(slice, [sub_i, sub_j], i,...
                    'Fontsize', 12, 'TextColor', colors(i, :) * 255, 'BoxOpacity', 0,...
                    'AnchorPoint', 'RightBottom'); % note this 255..
            catch
            end
        end
        img_valid(:, :, :, j) = slice;
        % for debugging
%         if j == 20
%         figure(101), imshow(slice), title(sprintf('component %d depth %d', i, j)), pause(0.01)
%         end
        
        % add boundary box to the original image
        bw_seg_bound = bw_seg - imerode(bw_seg, erode_se);
        real_silce = squeeze(valid_in_std_recon_img(:, :, :, j));
        real_silce = labeloverlay(real_silce, bw_seg_bound, 'colormap', colors(i, :));
        
        % add component number
        if ~isempty(ind)  
            try  % insertText() is in the Computer Vision Toolbox and might not be available
                real_silce = insertText(real_silce, [sub_i, sub_j], i,...
                    'Fontsize', 12, 'TextColor', colors(i, :) * 255, 'BoxOpacity', 0,...
                    'AnchorPoint', 'RightBottom'); % note this 255..
            catch
            end
        end
        valid_in_std_recon_img(:, :, :, j) = real_silce;
%         if j == 20
%         figure(102), imshow(real_silce), title(sprintf('component %d depth %d', i, j)), pause(0.01)
%         end  
    end
end

% spatial cut
if ~isempty(cut_box)
    % cut based on the top left corner
    ind = find(sum(img_valid, 3));
    [sub_h, sub_w, sub_z] = ind2sub([size_h, size_w, size_z], ind(1));
    % judge bounary
    h_start = max(sub_h - floor(cut_box(1) / 2), 1);
    h_end = min(sub_h + floor(cut_box(1) / 2), size_h);
    w_start = max(sub_w - floor(cut_box(2) / 2), 1);
    w_end = min(sub_w + floor(cut_box(2) / 2), size_w);
    z_start = max(sub_z - floor(cut_box(3) / 2), 1);
    z_end = min(sub_z + floor(cut_box(3) / 2), size_z);
    
    img_valid = img_valid(h_start: h_end, w_start: w_end, :, z_start: z_end);
    valid_in_std_recon_img = valid_in_std_recon_img(h_start: h_end, w_start: w_end,...
        :, z_start: z_end);
end
hyperstack_write(fullfile(output_dir, [name '_color_segment.tif']), img_valid);
hyperstack_write(fullfile(output_dir, [name '_std_img_segment.tif']), valid_in_std_recon_img);
end