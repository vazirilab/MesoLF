function lightfield_shifted = virtual_shift(lightfield_captured, px_per_ml, alpha_calibrated_array, source_plane_depth, target_depth)
lightfield_shifted = zeros(size(lightfield_captured, 1), size(lightfield_captured, 2));
% loop over sub-aperture indices (often denoted as u,v)
for i = 1 : px_per_ml
    for j = 1 : px_per_ml
        % collect pixels for current sub-aperture image
        buf = gather(lightfield_captured(i : px_per_ml : end, j : px_per_ml : end));
        % upsample sub-aperture image
        buf = imresize(buf, px_per_ml);
        
        % convert 1-based subaperture indices to indices relative to
        % central subaperture pixel
        curr_u = i - ceil(px_per_ml / 2);
        curr_v = j - ceil(px_per_ml / 2);
        
        % compute the shift that would need to be applied to lightfield_captured to refocus the
        % source plane (which is located at source_plane_depth in lightfield_captured)
        curr_shift_u = -curr_u * alpha_calibrated_array(source_plane_depth) * px_per_ml;
        curr_shift_v = -curr_v * alpha_calibrated_array(source_plane_depth) * px_per_ml;
        
        % compute the shift that would need to be applied to lightfield_captured to refocus the
        % target_depth (the depth from where the source plane should appear
        % to be located)
        target_shift_u = -curr_u * alpha_calibrated_array(target_depth) * px_per_ml;
        target_shift_v = -curr_v * alpha_calibrated_array(target_depth) * px_per_ml;
        
        % the difference between these two shift values is what we need to
        % apply to the current subaperture image
        mov_shift_u = target_shift_u - curr_shift_u;
        mov_shift_v = target_shift_v - curr_shift_v;
        
        % apply the shift
        img_shift = ImWarp(buf, -mov_shift_v, -mov_shift_u); % note the direction of x and y
        
        % downsample
        img_shift_d = imresize(img_shift, 1 / px_per_ml);
        
        % place subaperture pixels back into full lightfield image
        lightfield_shifted(i : px_per_ml : end, j : px_per_ml : end) = img_shift_d;       
    end
end
end
