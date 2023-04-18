function recon_stack_filtered = filter_with_bloodvessel_mask(recon_stack, SI)
% SI.vessel_thresh

symmfilter = struct();
symmfilter.sigma     = 2;
symmfilter.len       = 60;
symmfilter.sigma0    = 2;
symmfilter.alpha     = 0.7;
asymmfilter = false;

% Apple BCOSFIRE filter
response_stack = BCOSFIRE_lfm(recon_stack, symmfilter, asymmfilter);

% Threshold filter response and dilate a bit
se = strel('disk',4);
response_stack_segm = zeros(size(recon_stack));
for zix = 1:size(recon_stack,3)
    response_stack_segm(:,:,zix) = imdilate(response_stack(:,:,zix) > SI.vessel_thresh, se);
end
recon_stack_filtered = recon_stack .* abs(1 - response_stack_segm);

% Plot
figure('units','normalized','outerpos',[0 0 1 1]);
zix = 2;
n = 1;
subplot(3,4,n);
imagesc(squeeze(recon_stack(:,:,zix))); axis off; axis image; title('Input image');
n = n + 1;
subplot(3,4,n);
imagesc(squeeze(response_stack(:,:,zix))); axis off; axis image; title('B-COSFIRE response image');
n = n + 1;
subplot(3,4,n);
imagesc(squeeze(response_stack_segm(:,:,zix))); axis off; axis image; title('B-COSFIRE segmented image');
n = n + 1;
subplot(3,4,n);
imagesc(squeeze(recon_stack_filtered(:,:,zix))); axis off; axis image; title('Filtered image');

zix = 20;
n = n + 1;
subplot(3,4,n);
imagesc(squeeze(recon_stack(:,:,zix))); axis off; axis image; title('Input image');
n = n + 1;
subplot(3,4,n);
imagesc(squeeze(response_stack(:,:,zix))); axis off; axis image; title('B-COSFIRE response image');
n = n + 1;
subplot(3,4,n);
imagesc(squeeze(response_stack_segm(:,:,zix))); axis off; axis image; title('B-COSFIRE segmented image');
n = n + 1;
subplot(3,4,n);
imagesc(squeeze(recon_stack_filtered(:,:,zix))); axis off; axis image; title('Filtered image');


zix = 48;
n = n + 1;
subplot(3,4,n);
imagesc(squeeze(recon_stack(:,:,zix))); axis off; axis image; title('Input image');
n = n + 1;
subplot(3,4,n);
imagesc(squeeze(response_stack(:,:,zix))); axis off; axis image; title('B-COSFIRE response image');
n = n + 1;
subplot(3,4,n);
imagesc(squeeze(response_stack_segm(:,:,zix))); axis off; axis image; title('B-COSFIRE segmented image');
n = n + 1;
subplot(3,4,n);
imagesc(squeeze(recon_stack_filtered(:,:,zix))); axis off; axis image; title('Filtered image');

timestr = datestr(now, 'YYmmddTHHMM');
saveas(gca, fullfile(SI.outdir, [timestr '_vessels_mask.png']));
savefig(fullfile(SI.outdir, [timestr '_vessels_mask.fig']));