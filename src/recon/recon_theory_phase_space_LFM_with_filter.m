
%% here recorld all the functions that needs for phase space LFM reconstruction
%  this one is exactly what the theory suggested, but with additional
%  low-pass filter in the sample space

%  update: make sure before/after filtering, the totoal intensity is the
%  same
%
%  Last update: 9/23/2020
function Xguess = recon_theory_phase_space_LFM_with_filter(LFIMG, H, Ht, maxIter, gpuFlag, gpu_id)
%  change to phase space PSF

for i = 1 : size(H ,5) % normalizee the depth 
    H(:, :, :, :, i) = H(:, :, :, :, i) / sum(H(:, :, :, :, i), 'all');
    Ht(:, :, :, :, i) = Ht(:, :, :, :, i) / sum(Ht(:, :, :, :, i), 'all');
end
[H, Ht] = gen_phasespace_PSF_theory(H, Ht);
Nnum = size(H, 3);
%% Weights definition
weight_cut = 0.2;
omega_NIP= Nnum-7;
NIP_range = ceil(size(H, 5) / 3); % be even
if mod(NIP_range, 2) == 1
    NIP_range = NIP_range + 1; 
end
NIP_range_omega = 100;


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

%saveastiff(im2uint16(weight), 'weight.tiff')

%% filter bank
size_h = size(LFIMG, 1);
size_w = size(LFIMG, 2);
x= (1 : size_h) - size_h/2;
y= (1 : size_w) - size_w/2;
[yy, xx]=meshgrid(y, x);
Nnum = size(H, 3);
obj_space_ft_filter_bank = zeros(size_h, size_w, NIP_range * 3);
obj_space_filter_bank = zeros(size_h, size_w, NIP_range * 3);

NIP_range_axis = (1 : NIP_range * 3) -  NIP_range * 3 / 2; 
omega_NIP_array = omega_NIP * exp(-(NIP_range_axis / NIP_range_omega * 2 * sqrt(log(2))).^2);
omega_NIP_array(omega_NIP_array < 0.2) = 0.2;
for i = 1 : NIP_range * 3 % 3sigma rule
    obj_space_filter_bank(:, :, i) = exp(-((xx / omega_NIP_array(i)).^2+(yy / omega_NIP_array(i)).^2));
    buf= abs(fftshift(fft2(obj_space_filter_bank(:, :, i))));
    obj_space_ft_filter_bank(:, :, i) = buf / max(buf(:));
end
% figure, imshow(obj_space_ft_filter_bank(:, :, round(end / 2)), []), title('NIP aperture filter')
% figure, imshow(abs(fftshift(fft2(obj_space_ft_filter_bank(:, :,end / 2)))), []), title('mini spot in NIP')
% figure, plot(omega_NIP_array), title('minimal sturcture')
% saveastiff(im2uint16(obj_space_ft_filter_bank/max(obj_space_ft_filter_bank(:))), sprintf('%s\\obj_space_ft_filter_bank.tiff', output_path))

obj_space_filter_bank = obj_space_filter_bank(floor(end / 2) - 20 : floor(end / 2) + 20, floor(end / 2) - 20 : floor(end / 2) + 20, :);
%obj_space_filter_bank = obj_space_filter_bank(end / 2 - 20:end / 2 + 20, end / 2 - 20:end / 2 + 20, :);

%% define operator
% initialization value
Xguess = ones(size(LFIMG, 1), size(LFIMG, 2), size(H, 5));
Xguess = Xguess ./ sum(Xguess(:)) .* sum(LFIMG(:)); % this is a poor initialize

if gpuFlag
    gpu = gpuDevice(gpu_id);
    reset(gpu);
    xsize = [size(LFIMG, 1), size(LFIMG, 2)];
    msize = [size(H, 1), size(H, 2)];
    mmid = floor(msize / 2);
    exsize = xsize + mmid; % to make the size as 2^N after padding
    exsize = gpuArray([min(2^ceil(log2(exsize(1))), 128 * ceil(exsize(1) / 128)), min(2^ceil(log2(exsize(2))), 128 * ceil(exsize(2) / 128))]);
    zeroImageEx = gpuArray(zeros(exsize, 'double'));
    Xguess = gpuArray(Xguess);

    % TN 2021-07-21: these function handles cause weird "named symbol not found" CUDA errors, so just replacing them with explicit calls
    %backwardFUN = @(psft, projection, u, v, Nnum) backwardProjectGPU_phasespace_theory(gpuArray(psft), gpuArray(projection), zeroImageEx, gpuArray(exsize), gpuArray(u), gpuArray(v), gpuArray(Nnum));
    %forwardFUN = @(psf, Xguess, Nnum) forwardProjectGPU_phasespace_theory(gpuArray(psf), gpuArray(Xguess), zeroImageEx, gpuArray(exsize), gpuArray(Nnum)); % one use H and one use Ht
%else
    %forwardFUN = @(psf, Xguess, Nnum) forwardProjectACC_phasespace_theory(H, Xguess, Nnum); % build the function: forward and backward
    %backwardFUN = @(psft, projection, u, v, Nnum) backwardProject_phasespace_theory(psft, projection, u, v, Nnum);
end

% Generate "spiral" index list for subaperture image update sequence
[index1, index2] = gen_spiral(Nnum);

% main iteration
for i = 1:maxIter
    tic;
    for u_2 = 1:Nnum
        for v_2 = 1:Nnum
            u = index1((u_2 - 1) * Nnum + v_2); % marginal to center, rebundant information first
            v = index2((u_2 - 1) * Nnum + v_2);
            if weight(u, v) == 0 % update weight
                continue;
            else
                if gpuFlag == true
                   %forwardFUN(squeeze(H(:, :, u, v, :)), Xguess, Nnum); % Note HXguess would be small
                   HXguess = forwardProjectGPU_phasespace_theory(gpuArray(H(:, :, u, v, :)), Xguess, zeroImageEx, exsize, gpuArray(Nnum)); % one use H and one use Ht
                else
                   %forwardFUN(squeeze(H(:, :, u, v, :)), Xguess, Nnum); % Note HXguess would be small
                   HXguess = forwardProjectACC_phasespace_theory(squeeze(H(:, :, u, v, :)), Xguessv, Nnum); 
                end
                %                 figure, imagesc(HXguess)
                % selective
                HXguess = squeeze(LFIMG(u:Nnum:end, v:Nnum:end)) ./ HXguess;
                HXguess(~isfinite(HXguess)) = 0;
                
                if gpuFlag == true
                    %XguessCor = backwardFUN(squeeze(Ht(:, :, u, v, :)), HXguess, u, v, Nnum);
                    XguessCor = backwardProjectGPU_phasespace_theory(gpuArray(squeeze(Ht(:, :, u, v, :))), HXguess, zeroImageEx, exsize, gpuArray(u), gpuArray(v), gpuArray(Nnum));
                    %buf = backwardFUN(squeeze(Ht(:, :, u, v, :)), ones(size(LFIMG) / Nnum), u, v, Nnum); % without pre-computing, why this?
                    buf = backwardProjectGPU_phasespace_theory(gpuArray(squeeze(Ht(:, :, u, v, :))), gpuArray(ones(size(LFIMG) / Nnum)), zeroImageEx, exsize, gpuArray(u), gpuArray(v), gpuArray(Nnum));
                else
                    %XguessCor = backwardFUN(squeeze(Ht(:, :, u, v, :)), HXguess, u, v, Nnum);
                    XguessCor = backwardProject_phasespace_theory(squeeze(Ht(:, :, u, v, :)), HXguess, u, v, Nnum);
                    %buf = backwardFUN(squeeze(Ht(:, :, u, v, :)), ones(size(LFIMG) / Nnum), u, v, Nnum); % without pre-computing, why this?
                    buf = backwardProject_phasespace_theory(squeeze(Ht(:, :, u, v, :)), ones(size(LFIMG) / Nnum), u, v, Nnum);
                end
                XguessCor = Xguess .* XguessCor ./ buf;

                XguessCor(isnan(XguessCor)) = 0;
                XguessCor(isinf(XguessCor)) = 0;
                XguessCor(XguessCor < 0) = 0;
                % update
                Xguess = XguessCor .* weight(u, v) + (1 - weight(u, v)) .* Xguess;
                Xguess(Xguess < 0) = 0;
            end
        end
    end
    % overal filter
    Xguess = sample_filter(Xguess, obj_space_ft_filter_bank);

    ttime1 = toc;
    disp(['iter ', num2str(i), ' | ', num2str(maxIter), ', phase-space deconvolution took ', num2str(ttime1), ' secs']);
end
Xguess = gather(Xguess);
if gpuFlag
    gpuDevice([]);
end
end

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
end

%% forward projection function
function projection_out = forwardProjectACC_phasespace_theory(H, realspace, Nnum)
% build forward function
projection = zeros(size(realspace, 1), size(realspace, 2));
for z = 1:size(realspace, 3)
    projection = projection + conv2(realspace(:, :, z), H(:, :, z), 'same');
end
projection_out = projection((Nnum + 1) / 2:Nnum:end, (Nnum + 1) / 2:Nnum:end);
end

%% backprojection function
function Backprojection = backwardProject_phasespace_theory(Ht, projection, u, v, Nnum)
big_projection = zeros(Nnum * size(projection));
big_projection(u:Nnum:end, v:Nnum:end) = projection;
Backprojection = zeros(size(big_projection, 1), size(big_projection, 2), size(Ht, 3));
for z = 1:size(Ht, 3)
    Backprojection(:, :, z) = conv2(big_projection, Ht(:, :, z), 'same');
end
end

%% backprojection function with GPU
function Backprojection = backwardProjectGPU_phasespace_theory(Ht, projection, zeroImageEx, exsize, u, v, Nnum)
% generate backward projection
big_projection = gpuArray.zeros(Nnum * size(projection));
big_projection(u:Nnum:end, v:Nnum:end) = projection;
Backprojection = gpuArray.zeros(size(big_projection, 1), size(big_projection, 2), size(Ht, 3));

for z = 1:size(Ht, 3)
    Backprojection(:, :, z) = conv2FFT(big_projection, Ht(:, :, z), zeroImageEx, exsize);
end
end

function out_sample = sample_filter(sample, obj_space_ft_filter_bank)
out_sample = gpuArray.zeros(size(sample, 1), size(sample, 2), size(sample, 3));
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
end

%% forward function with GPU
function projection_out = forwardProjectGPU_phasespace_theory(H, realspace, zeroImageEx, exsize, Nnum)
projection = gpuArray.zeros(size(realspace, 1), size(realspace, 2));
for z = 1:size(realspace, 3)
    projection = projection + conv2FFT(realspace(:, :, z), H(:, :, z), zeroImageEx, exsize);
end
projection_out = projection((Nnum + 1) / 2:Nnum:end, (Nnum + 1) / 2:Nnum:end); % additional selection
end

function out_img = freq_filter(image, filter)
image_f = fftshift(fft2(image));
out_img = real(ifft2(ifftshift(image_f .* filter)));
end
