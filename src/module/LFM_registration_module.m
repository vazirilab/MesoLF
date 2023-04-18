function sensor_movie = LFM_registration_module(sensor_movie, patch_info, SI)

%% module for LFM registration.
%   Input
%   sensor_movie: xyt sensor image
%   patch_info: contains size info of the sensor_movie
%   SI: configuration structure
%       SI.bin_width: registration feature size

%   Output
%   sensor_movie: registered sensor_movie.

%   last update: 7/7/2020. YZ


%% parser
Nnum = SI.Nnum;
bin_width = SI.reg_bin_width;
outdir = SI.outdir;

size_h = patch_info.size(1);
size_w = patch_info.size(2);
if length(size(sensor_movie)) == 2
    frameN = size(sensor_movie, 2);
elseif length(size(sensor_movie)) == 3
    frameN = size(sensor_movie, 3);
end
%% reconstruction
disp('Compute LFM motion correction shifts')
sensor_movie = reshape(sensor_movie, size_h, size_w, frameN);

central_img = sensor_movie(ceil(Nnum / 2) : Nnum : end, ceil(Nnum / 2) : Nnum : end, :);

[d1,d2, ~] = size(central_img);
[~, shifts, bound, option_nr] = motion_correction(central_img, d1, d2, bin_width, outdir);
save(fullfile(SI.outdir, [datestr(now, 'YYmmddTHHMM') '_motion_correction.mat']), 'shifts', 'bound', 'option_nr', '-v7.3');
new_sensor_img = zeros(size_h, size_w, frameN);

textprogressbar('Apply LFM motion correction shifts')
for i = 1 : Nnum
    for j = 1 : Nnum
        % progress bar     
        textprogressbar(((i - 1) * Nnum + j) / Nnum / Nnum * 100)
        buf = sensor_movie(i : Nnum : end, j : Nnum : end, :);
        new_sensor_img(i : Nnum : end, j : Nnum : end, :) = ...
            apply_shifts(buf, shifts, option_nr, bound/2, bound/2);        
    end
end
sensor_movie = new_sensor_img;
clear new_sensor_img
textprogressbar('done')

sensor_movie = reshape(sensor_movie, [], frameN);
end
