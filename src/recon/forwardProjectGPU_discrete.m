function TOTALprojection = forwardProjectGPU_discrete(H, realspace, zeroImageEx, exsize, unique_array, max_PSF_size)
Nnum = size(H,3);
zerospace = gpuArray.zeros(size(realspace,1), size(realspace,2), 'single');
TOTALprojection = gpuArray.zeros(size(realspace,1), size(realspace,2), 'single');
PSF_size_half = floor(max_PSF_size / 2);
for aa=1:Nnum
    for bb=1:Nnum
        for cc=1:size(realspace,3)
            Hs = gpuArray(squeeze(H(:, :, aa, bb, unique_array(cc))));
            % crop
            Hs_crop = Hs(ceil(end / 2) - PSF_size_half : ceil(end / 2) + PSF_size_half, ...
                         ceil(end / 2) - PSF_size_half : ceil(end / 2) + PSF_size_half);
            tempspace = gpuArray.zeros(size(realspace,1), size(realspace,2), 'single');
            tempspace(aa:Nnum:end, bb:Nnum:end) = realspace(aa:Nnum:end, bb:Nnum:end, cc);
            projection = conv2FFT(tempspace, Hs_crop, zeroImageEx, exsize);
            TOTALprojection = TOTALprojection + projection;
        end
    end
end
TOTALprojection = double(TOTALprojection);
