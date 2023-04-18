function TOTALprojection = forwardProjectACC( H, realspace, CAindex)
% build forward function
Nnum = size(H,3);
zerospace = zeros(  size(realspace,1),   size(realspace,2), 'single');
TOTALprojection = zerospace;
% just loop for a small space behind a microlens
for aa=1:Nnum
    for bb=1:Nnum
        for cc=1:size(realspace,3) % for each spatial points

            % read out the PSF
            Hs = squeeze(H( CAindex(cc,1):CAindex(cc,2), CAindex(cc,1):CAindex(cc,2) ,aa,bb,cc));          
            tempspace = zerospace;
            tempspace( (aa:Nnum:end), (bb:Nnum:end) ) = realspace( (aa:Nnum:end), (bb:Nnum:end), cc); % get the pixels from the same place out
            projection = conv2(tempspace, Hs, 'same');
            TOTALprojection = TOTALprojection + projection;            
        end
    end
end