function TOTALprojection = forwardProjectGPU_skip_2D( H, realspace, zeroImageEx, exsize, step, cc)

Nnum = size(H,3);
zerospace = gpuArray.zeros(  size(realspace,1),   size(realspace,2), 'single');
TOTALprojection = zerospace;

for aa=1:step:Nnum
    for bb=1:step:Nnum
        Hs = gpuArray(squeeze(H( :,:,aa,bb,cc)));    
        tempspace = zerospace;
        tempspace( (aa:Nnum:end), (bb:Nnum:end) ) = realspace( (aa:Nnum:end), (bb:Nnum:end));
        projection = conv2FFT(tempspace, Hs, zeroImageEx, exsize);
        TOTALprojection = TOTALprojection + projection;            
    end
end
TOTALprojection = double(TOTALprojection);
