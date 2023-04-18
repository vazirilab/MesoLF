%% Build
mexcuda mex_lfm_convolution_vec.cu

%% Build with debug info
mexcuda -G -dynamic mex_lfm_convolution_3d.cu  % -G = debug info

%%
% feval('_gpu_asyncState', false);
% feval('_gpu_setLazyEvaluationEnabled', false);
% feature('GpuAllocPoolSizeKb', 0);

%%
% % C-order = row-major order: when going through mem, column index varies
% % fastest
% index = (r * nbr_of_cols) + c;
% 
% % F-order = column-major order: when going through mem, row index varies
% % fastest
% index = r + (c * nbr_of_rows);
% 
% % juro accesses the image like this:
% output[x + y * bitmap_width] = value;
% % so he expects C-order, with indices in dequence (row, column)
% 
% % he goes through the kernels like this:
% int kernel_offset = max_kernel_width * max_kernel_width * current_kernel +
%                       max_kernel_width * kernel_starty +
%                       kernel_starti;
% %his docs: "dev_kernels: num_kernels x num_kernels consecutive kernels, each
% %with max_kernel_width x max_kernel_width pixels."
% 
% % so he expects C-order, with index sequence: (kernel_ix_x, kernel_ix_y, px_in_kernel_x, px_in_kernel_y)
% 
% % after conversion using c2f(), a matrix will have C-order in mem, but
% % Matlab will still give the original (matlab dimensions). so in the host
% % code, we keep the order in which matlab reports them

%%
gpu = gpuDevice(1);

%% Try using juro's conv. for forward projection
vols = zeros(20*15, 21*15, 10, 61, 'single');
vols(1:30, 1:30, :, 25:26) = 1;

%%
%psf = load('/data0/tn/wisim/psf/PSFmatrix_lfm2pram_M10_FN12p5_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat');
% psf = PSF_struct;
psf.H = H;
psf.Ht = Ht;
%% Regular forward-project of one vol
vol = squeeze(vols(:,:,1,:));
vol_gpu = gpuArray(vol);
psf_gpu = gpuArray(psf.H);

tic;
proj_ref = gather(forwardProjectGPU(psf_gpu, vol_gpu, gpuArray(zeros(size(vol, 1:2))), size(vol, 1:2)));
toc;

%% 
psf_gpu = gpuArray(psf.Ht);
proj_juro = gpuArray(zeros(size(vols, 1), size(vols, 2), size(vols, 3),'single'));
proj_juro= padarray(proj_juro, [90, 90], 'post');
vols_gpu = gpuArray(vols);
tic;
for zi = 1 : size(psf_gpu, 5)
    psf_slice = squeeze(psf_gpu(:,:,:,:,zi));
    vols_slice = squeeze(vols_gpu(:,:,:,zi)); % third index is no. of image to project
    
    % if pad
    vols_slice = padarray(vols_slice, [90, 90], 'post');
    
    proj = mex_lfm_convolution_vec(vols_slice, psf_slice);
    proj_juro = proj_juro + proj;
end
%clear proj_gpu;
%proj_gpu = gpuArray(zeros(size(img_tmp), 'single'));
%proj_gpu = mex_lfm_convolution_noop(img_gpu, psf_gpu);
toc
figure
imagesc(squeeze(proj_juro(:,:,1)));
%% Compare
figure;
subplot(131);
imagesc(proj_ref);
axis image;
colorbar;
subplot(132);
imagesc(squeeze(proj_juro(:,:,1)));
axis image;
colorbar;
% subplot(133);
% imagesc(squeeze(proj_juro(1,:,:)) - proj_ref);
% axis image;
% colorbar;

%%
gpuDevice([]);
