function [H_deshift, Ht] = gen_phasespace_PSF_theory(H, Ht)

%% Note the generation of Ht is different from phase PSF
Nnum = size(H, 3);
% Realignment H to phase_space PSF
IMGsize = size(H, 1) - mod((size(H, 1) - Nnum), 2 * Nnum); % make sure there is odd number of MLA pieces in PSF
index1 = round(size(H, 1) / 2) - fix(IMGsize / 2); % this is a cut boundary
index2 = round(size(H, 1) / 2) + fix(IMGsize / 2);
H_deshift = zeros(IMGsize, IMGsize, Nnum, Nnum, size(H, 5));
for z = 1:size(H, 5) % for each depth
    LFtmp = zeros(IMGsize, IMGsize, Nnum, Nnum);
    % for each position
    for ii = 1:size(H, 3) % for N_num
        for jj = 1:size(H, 4) % for N_num
            LFtmp(:, :, ii, jj) = im_shift3(squeeze(H(index1:index2, index1:index2, ii, jj, z)), ...
                ii - (Nnum + 1) / 2, ...
                jj - (Nnum + 1) / 2); % shift??? original PSF generation has a shift
        end
    end
    H_deshift(:, :, :, :, z) = LFtmp;
end
H_deshift_sz = size(H_deshift);
H_deshift = reshape(H_deshift, Nnum, H_deshift_sz(1) / Nnum, Nnum, H_deshift_sz(1) / Nnum, Nnum, Nnum, H_deshift_sz(5));
H_deshift = permute(H_deshift, [1, 3, 2, 4, 5, 6, 7]); % Nnum, Nnum, Nmla, Nmla, Nnum, Nnum, z
H_deshift = H_deshift(:, :, :, :, end:-1:1, end:-1:1, :);
H_deshift = permute(H_deshift, [1, 2, 5, 3, 6, 4, 7]); % Nnum, Nnum, Nnum, Nmla, Nnum, Nmla, z
H_deshift = permute(reshape(H_deshift, Nnum, Nnum, H_deshift_sz(1), H_deshift_sz(1), H_deshift_sz(5)), [3, 4, 1, 2, 5]);
% additional shift, since convolution involved

% generate psf t
Ht = Ht(index1:index2, index1:index2, :, :, :);

%% utility function
function new_im = im_shift3(img, SHIFTX, SHIFTY)
eqtol = 1e-10;

xlength = size(img, 1);
ylength = size(img, 2);

if abs(mod(SHIFTX, 1)) > eqtol || abs(mod(SHIFTY, 1)) > eqtol
    error('SHIFTX and SHIFTY should be integer numbers');
end

SHIFTX = round(SHIFTX);
SHIFTY = round(SHIFTY);

new_im = zeros(xlength, ylength, size(img, 3));

if SHIFTX >= 0 && SHIFTY >= 0
    new_im((1 + SHIFTX:end), (1 + SHIFTY:end), :) = img((1:end - SHIFTX), (1:end - SHIFTY), :);
elseif SHIFTX >= 0 && SHIFTY < 0
    new_im((1 + SHIFTX:end), (1:end + SHIFTY), :) = img((1:end - SHIFTX), (-SHIFTY + 1:end), :);
elseif SHIFTX < 0 && SHIFTY >= 0
    new_im((1:end + SHIFTX), (1 + SHIFTY:end), :) = img((-SHIFTX + 1:end), (1:end - SHIFTY), :);
else
    new_im((1:end + SHIFTX), (1:end + SHIFTY), :) = img((-SHIFTX + 1:end), (-SHIFTY + 1:end), :);
end
