function TOTALprojection = forwardProjectGPU( H, realspace, zeroImageEx, exsize)

Nnum = size(H,3);
zerospace = gpuArray.zeros(  size(realspace,1),   size(realspace,2), 'single');
TOTALprojection = zerospace;

for aa=1:Nnum
    for bb=1:Nnum
        for cc=1:size(realspace,3),
    
            Hs = gpuArray(squeeze(H( :,:,aa,bb,cc)));    
            tempspace = zerospace;
            tempspace( (aa:Nnum:end), (bb:Nnum:end) ) = realspace( (aa:Nnum:end), (bb:Nnum:end), cc);
            projection = conv2FFT(tempspace, Hs, zeroImageEx, exsize);
            TOTALprojection = TOTALprojection + projection;            

        end
    end
end
TOTALprojection = double(TOTALprojection);
