
%% Phase space LFM reconstruction with simple peeling.
%  LFM part:
%  	this one is exactly what the theory suggested, but with additional
%  	low-pass filter in the sample space
%  peeling part:
%   this one is a simple top-layer peeling method, which suggests the top layer would contaminate
%   reconstructions of following layers

%  update: put the filter at the loop.
%          only remove the top layer
%  Last update: 4/4/2020
function [Xguess, LFIMG, proj_sensor] = recon_theory_phase_space_LFM_with_filter_simple_peeling(LFIMG, ...
    H_origin, Ht_origin, maxIter, gpuFlag, gpu_id)
if nargin < 6
    gpu_id = 1;
end
if nargin < 5
    gpu_id = false;
    gpuFlag = false;
end
for i = 1 : size(H ,5) % normalizee the depth 
    H_origin(:, :, :, :, i) = H_origin(:, :, :, :, i) / sum(H_origin(:, :, :, :, i), 'all');
    Ht_origin(:, :, :, :, i) = Ht_origin(:, :, :, :, i) / sum(Ht_origin(:, :, :, :, i), 'all');
end
[H, Ht] = gen_phasespace_PSF_theory(H_origin, Ht_origin);
Nnum = size(H, 3);

%% parameters
omega_NIP = Nnum - 7;
NIP_range_omega = 100;
peeling_mask_ratio = 0.1;
reduce_ratio = 0.9;

%% Weights definition
% Weights for every iteration
weight=squeeze(sum(sum(sum(H,1),2),5)); % jsut 2d, ummmm
weight(find(isnan(weight))) = 0;
weight = weight./sum(weight(:)); % sum
weight = weight-min(weight(:)); % since we minus the minimum
% weight(weight<0.03)=0;
% weight=weight.*80;
weight = weight / max(weight(:)) ;
weight(weight<weight_cut)=0;
weight = weight * 0.3;
% weight(weight>0) = 0.4;
max_weight = max(weight(:));
% saveastiff(im2uint16(weight), 'weight.tiff')

%% filter bank
x = (1:size(LFIMG, 1)) - size(LFIMG, 1) / 2;
y = (1:size(LFIMG, 2)) - size(LFIMG, 2) / 2;
[yy, xx] = meshgrid(y, x);

NIP_range = floor(size(H, 5) / 3);
if mod(NIP_range, 2) == 1 % keep even
    NIP_range = NIP_range - 1;
end

obj_space_ft_filter_bank = zeros(size(LFIMG, 1), size(LFIMG, 2), NIP_range * 3);
obj_space_filter_bank = zeros(size(LFIMG, 1), size(LFIMG, 2), NIP_range * 3);

NIP_range_axis = (1:NIP_range * 3) - NIP_range * 3 / 2;

omega_NIP_array = omega_NIP * exp(-(NIP_range_axis / NIP_range_omega * 2 * sqrt(log(2))).^2);
omega_NIP_array(omega_NIP_array < 0.2) = 0.2;
for i = 1:NIP_range * 3 % 3sigma rule
    obj_space_filter_bank(:, :, i) = exp(-((xx / omega_NIP_array(i)).^2 + (yy / omega_NIP_array(i)).^2));
    buf = abs(fftshift(fft2(obj_space_filter_bank(:, :, i))));
    obj_space_ft_filter_bank(:, :, i) = buf / max(buf(:));
end
obj_space_filter_bank = obj_space_filter_bank(end / 2 - 20:end / 2 + 20, end / 2 - 20:end / 2 + 20, :);

%% define operators
if gpuFlag == true
    gpu = gpuDevice(gpu_id); %#ok<NASGU>
    xsize = [size(LFIMG, 1), size(LFIMG, 2)];
    msize = [size(H, 1), size(H, 2)];
    mmid = floor(msize / 2);
    exsize = xsize + mmid; % to make the size as 2^N after padding
    exsize = [min(2^ceil(log2(exsize(1))), 128 * ceil(exsize(1) / 128)), min(2^ceil(log2(exsize(2))), 128 * ceil(exsize(2) / 128))];
    zeroImageEx = gpuArray(zeros(exsize, 'single'));
    % push to GPU
    %     Ht = gpuArray(Ht);
    %     H = gpuArray(H);
    %     LFIMG = gpuArray(LFIMG);

    backwardFUN = @(psft, projection, u, v, Nnum) backwardProjectGPU_phasespace_theory(psft, gpuArray(projection), zeroImageEx, exsize, u, v, Nnum);
    forwardFUN = @(psf, Xguess, Nnum) forwardProjectGPU_phasespace_theory(psf, gpuArray(Xguess), zeroImageEx, exsize, Nnum); % one use H and one use Ht
else
    forwardFUN = @(psf, Xguess, Nnum) forwardProjectACC_phasespace_theory(H, Xguessv, Nnum); % build the function: forward and backward
    backwardFUN = @(psft, projection, u, v, Nnum) backwardProject_phasespace_theory(psft, projection, u, v, Nnum);
end

% Xguess = contrastAdjust(Xguess, contrast);

% update index
[index1, index2] = gen_spiral(Nnum);

% generate Htf
% Htf = zeros(size(LFIMG,1),size(LFIMG,2), size(H,5), Nnum ,Nnum);
% for u=1:Nnum
%     for v=1:Nnum
%         if weight(u,v)==0
%             continue;
%         else
%             Htf(:,:,:,u,v)= backwardFUN(squeeze(Ht(:,:,u,v,:)), gpuArray(ones(size(LFIMG))));
%         end
%     end
% end

% if gpu_id ~= false
%     n_wait = 0;
%     while (gpu.AvailableMemory < whos('H').bytes) && (n_wait < 15)
%         disp('Pausing 60 s to wait for GPU to become available ');
%         pause(60);
%         n_wait = n_wait + 1;
%     end
% end  

%% first iteration
n_try = 0;
while true
    n_try = n_try + 1;
    try
        % initialization
        LFIMG = gpuArray(single(LFIMG));
        Xguess = gpuArray(ones(size(LFIMG, 1), size(LFIMG, 2), size(H, 5), 'single'));
        Xguess = Xguess ./ sum(Xguess(:)) .* sum(LFIMG(:)); % this is a poor initialize
        for i = 1:maxIter
            tic;
            for u_2 = 1:Nnum
                for v_2 = 1:Nnum
                    u = index1((u_2 - 1) * Nnum + v_2); % marginal to center, rebundant information first
                    v = index2((u_2 - 1) * Nnum + v_2);
                    %             u = u_2;
                    %             v = v_2;
                    if weight(u, v) == 0 % update weight
                        continue;
                    else
                        HXguess = forwardFUN(gpuArray(squeeze(H(:, :, u, v, :))), Xguess, Nnum); % Note HXguess would be small
                        % figure, imagesc(HXguess)
                        % selective
                        HXguess = squeeze(LFIMG(u:Nnum:end, v:Nnum:end)) ./ (HXguess);
                        HXguess(~isfinite(HXguess)) = 0;
                        
                        XguessCor = backwardFUN(gpuArray(squeeze(Ht(:, :, u, v, :))), HXguess, u, v, Nnum);
                        buf = backwardFUN(gpuArray(squeeze(Ht(:, :, u, v, :))), gpuArray(ones(size(LFIMG) / Nnum)), u, v, Nnum); % without pre-computing, why this?
                        XguessCor = Xguess .* XguessCor ./ (buf + 1e-10);
                        % XguessCor=Xguess.*XguessCor;
                        XguessCor(isnan(XguessCor)) = 0;
                        XguessCor(isinf(XguessCor)) = 0;
                        XguessCor(XguessCor < 0) = 0;
                        % update
                        Xguess = XguessCor .* weight(u, v) + (1 - weight(u, v)) .* Xguess;
                        Xguess(Xguess < 0) = 0;
                    end
                end
            end
            Xguess = sample_filter(Xguess, obj_space_ft_filter_bank, obj_space_filter_bank);
            
            ttime1 = toc;
            disp(['iter ', num2str(i), ' | ', num2str(maxIter), ', phase-space deconvolution took ', num2str(ttime1), ' secs']);
        end
        Xguess = gather(Xguess);
        clear HXguess XguessCor buf;
        break
    catch ME
        disp(ME);
        if n_try < 120
            disp('Something went wrong. Will pause 60s and re-try.');
            disp(['No. of tries so far: ' num2str(n_try)]);
            pause(60);
        else
            error('Too many re-tries. Giving up.')
        end  
    end
end
       

%% remove the top layer and run the second
disp('Starting peeling')
top_layer = Xguess(:, :, 1);
dummy_vol = gpuArray(zeros(size(Xguess), 'single'));
dummy_vol(:, :, 1) = top_layer;

% mask
dummy_mask = gpuArray(zeros(size(Xguess)));
dummy_mask(:, :, 1) = top_layer > max(top_layer(:)) * peeling_mask_ratio;

proj_sensor = forwardProjectGPU(H_origin, dummy_vol, zeroImageEx, exsize);
clear dummy_vol;
proj_sensor_mask = forwardProjectGPU(H_origin, dummy_mask, zeroImageEx, exsize);
clear dummy_mask;
proj_sensor_mask = proj_sensor_mask / sum(proj_sensor_mask(:));

% normalize
proj_sensor = proj_sensor * sum(proj_sensor_mask .* LFIMG, 'all') / ...
    sum(proj_sensor_mask .* proj_sensor, 'all');
clear proj_sensor_mask;

% peel
LFIMG = LFIMG - proj_sensor * reduce_ratio;
LFIMG(LFIMG < 0) = 0;
proj_sensor = gather(proj_sensor);

%% second run
n_try = 0;
while true
    n_try = n_try + 1;
    try
        Xguess = gpuArray(ones(size(LFIMG, 1), size(LFIMG, 2), size(H_origin, 5), 'single'));
        Xguess = Xguess ./ sum(Xguess(:)) .* sum(LFIMG(:)); % this is a poor initialize
        for i = 1:maxIter
            tic;
            for u_2 = 1:Nnum
                for v_2 = 1:Nnum
                    u = index1((u_2 - 1) * Nnum + v_2); % marginal to center, rebundant information first
                    v = index2((u_2 - 1) * Nnum + v_2);
                    %             u = u_2;
                    %             v = v_2;
                    if weight(u, v) == 0 % update weight
                        continue;
                    else
                        HXguess = forwardFUN(gpuArray(squeeze(H(:, :, u, v, :))), Xguess, Nnum); % Note HXguess would be small
                        %                 figure, imagesc(HXguess)
                        % selective
                        HXguess = squeeze(LFIMG(u:Nnum:end, v:Nnum:end)) ./ (HXguess);
                        HXguess(~isfinite(HXguess)) = 0;
                        
                        XguessCor = backwardFUN(gpuArray(squeeze(Ht(:, :, u, v, :))), HXguess, u, v, Nnum);
                        buf = backwardFUN(gpuArray(squeeze(Ht(:, :, u, v, :))), gpuArray(ones(size(LFIMG) / Nnum)), u, v, Nnum); % without pre-computing, why this?
                        XguessCor = Xguess .* XguessCor ./ (buf + 1e-10);
                        %                 XguessCor=Xguess.*XguessCor;
                        XguessCor(isnan(XguessCor)) = 0;
                        XguessCor(isinf(XguessCor)) = 0;
                        XguessCor(XguessCor < 0) = 0;
                        % update
                        Xguess = XguessCor .* weight(u, v) + (1 - weight(u, v)) .* Xguess;
                        Xguess(Xguess < 0) = 0;
                        
                        % overal filter
                    end
                end
            end
            Xguess = sample_filter(Xguess, obj_space_ft_filter_bank, obj_space_filter_bank);
            
            ttime1 = toc;
            disp(['iter ', num2str(i), ' | ', num2str(maxIter), ', phase-space deconvolution took ', num2str(ttime1), ' secs']);
        end
        break
    catch ME
        disp(ME);
        if n_try < 120
            disp('Something went wrong. Will pause 60s and re-try.');
            disp(['No. of tries so far: ' num2str(n_try)]);
            pause(60);
        else
            error('Too many re-tries. Giving up.')
        end  
    end
end
clear H Ht;
LFIMG = double(gather(LFIMG));
Xguess = double(gather(Xguess));
Xguess(:, :, 1) = top_layer;

%% utility functions

%% generate spiral index
function [i_index, j_index] = gen_spiral(Nnum)
if mode(Nnum) == 0
    loop_n = Nnum / 2;
else
    loop_n = (Nnum + 1) / 2;
end
i_index = zeros(1, Nnum * Nnum);
j_index = zeros(1, Nnum * Nnum);

start = 1;
for k = 1:loop_n
    current_size = Nnum - 2 * (k - 1);
    bias = k - 1;
    if current_size > 1
        i_new = [1 : current_size, current_size * ones(1, current_size - 1), ...
            current_size - 1 : -1 : 1, ones(1, current_size - 2)];

        j_new = [ones(1, current_size), 2 : current_size, ...
            current_size * ones(1, current_size - 1), current_size - 1 : -1 : 2];

        i_index(start:start + 4 * (current_size - 1) - 1) = i_new + bias;
        j_index(start:start + 4 * (current_size - 1) - 1) = j_new + bias;
        start = start + 4 * (current_size - 1);
    else
        i_new = 1;
        j_new = 1;
        i_index(start) = i_new + bias;
        j_index(start) = j_new + bias;
    end

end

%% forward projection function
function projection_out = forwardProjectACC_phasespace_theory(H, realspace, Nnum)
% build forward function
projection = zeros(size(realspace, 1), size(realspace, 2), 'single');
for z = 1:size(realspace, 3)
    projection = projection + conv2(realspace(:, :, z), H(:, :, z), 'same');
end
projection_out = projection((Nnum + 1) / 2:Nnum:end, (Nnum + 1) / 2:Nnum:end);

%% backprojection function
function Backprojection = backwardProject_phasespace_theory(Ht, projection, u, v, Nnum)
big_projection = zeros(Nnum * size(projection), 'single');
big_projection(u:Nnum:end, v:Nnum:end) = projection;
Backprojection = zeros(size(big_projection, 1), size(big_projection, 2), size(Ht, 3), 'single');
for z = 1:size(Ht, 3)
    Backprojection(:, :, z) = conv2(big_projection, Ht(:, :, z), 'same');
end

%% backprojection function with GPU
function Backprojection = backwardProjectGPU_phasespace_theory(Ht, projection, zeroImageEx, exsize, u, v, Nnum)
% generate backward projection
big_projection = gpuArray(zeros(Nnum * size(projection), 'single'));
big_projection(u:Nnum:end, v:Nnum:end) = projection;
Backprojection = gpuArray(zeros(size(big_projection, 1), size(big_projection, 2), size(Ht, 3), 'single'));

for z = 1:size(Ht, 3)
    Backprojection(:, :, z) = conv2FFT(big_projection, Ht(:, :, z), zeroImageEx, exsize);
end

function out_sample = sample_filter(sample, obj_space_ft_filter_bank, ~)
out_sample = gpuArray(zeros(size(sample, 1), size(sample, 2), size(sample, 3), 'single'));
half_z = round(size(sample, 3) / 2);
for z = 1:size(sample, 3)
    img = sample(:, :, z);
    if z >= half_z - size(obj_space_ft_filter_bank, 3) / 2 + 1 && ...
            z <= half_z + size(obj_space_ft_filter_bank, 3) / 2 % under the assumption of both the size is even
        % low pass
        sum_before = sum(img(:));
        buf = freq_filter(img, obj_space_ft_filter_bank(:, :, z - half_z + size(obj_space_ft_filter_bank, 3) / 2));
        buf = sum_before * buf / sum(buf(:));
        % deconv
        %         buf = deconvlucy(gather(buf), obj_space_filter_bank(:, :, z - (size(sample,3) - size(obj_space_filter_bank, 3))/2), 10);
        out_sample(:, :, z) = gpuArray(buf);
    else
        out_sample(:, :, z) = img;
    end
end

%% forward function with GPU
function projection_out = forwardProjectGPU_phasespace_theory(H, realspace, zeroImageEx, exsize, Nnum)
projection = gpuArray(zeros(size(realspace, 1), size(realspace, 2), 'single'));
for z = 1:size(realspace, 3)
    projection = projection + conv2FFT(realspace(:, :, z), H(:, :, z), zeroImageEx, exsize);
end
projection_out = projection((Nnum + 1) / 2:Nnum:end, (Nnum + 1) / 2:Nnum:end); % additional selection

function out_img = freq_filter(image, filter)
% note here filter is a frequency filter
image_f = fftshift(fft2(image));
out_img = real(ifft2(ifftshift(image_f .* filter)));

function TOTALprojection = forwardProjectGPU(H, realspace, zeroImageEx, exsize)
Nnum = size(H, 3);
zerospace = gpuArray(zeros(size(realspace, 1), size(realspace, 2), 'single'));
TOTALprojection = zerospace;

for aa = 1:Nnum
    for bb = 1:Nnum
        for cc = 1:size(realspace, 3)
            Hs = gpuArray(squeeze(H(:, :, aa, bb, cc)));
            tempspace = zerospace;
            tempspace((aa:Nnum:end), (bb:Nnum:end)) = realspace((aa:Nnum:end), (bb:Nnum:end), cc);
            projection = conv2FFT(tempspace, Hs, zeroImageEx, exsize);
            TOTALprojection = TOTALprojection + projection;

        end
    end
end
%TOTALprojection = double(TOTALprojection);
