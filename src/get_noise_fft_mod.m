function [sn,PSDX,ff] = get_noise_fft_mod(Y, options)
% return power spectrum of Y, as psdx
% Y has to be (n_pixels x n_timesteps)
% SN: average of signal in spectrum, within a window
% ff: spectrum coordinate
% based on https://github.com/flatironinstitute/CaImAn-MATLAB/blob/master/utilities/get_noise_fft.m
% Slightly modified and commented by Yz, last update: 2/16/2020

defoptions = struct;
defoptions.noise_range = [0.25,0.7];
defoptions.noise_method = 'logmexp';
defoptions.block_size = [64,64];
defoptions.split_data = false;
defoptions.max_timesteps = 3000;

if nargin < 2 || isempty(options); options = defoptions; end

% options
if ~isfield(options,'noise_range'); options.noise_range = defoptions.noise_range; end
range_ff = options.noise_range;
if ~isfield(options,'noise_method'); options.noise_method = defoptions.noise_method; end
method = options.noise_method;
if ~isfield(options,'block_size'); options.block_size = defoptions.block_size; end
block_size = options.block_size;
if ~isfield(options,'split_data'); options.split_data = defoptions.split_data; end
split_data = options.split_data;
if ~isfield(options,'max_timesteps') || isempty(options.max_timesteps)
    options.max_timesteps = defoptions.max_timesteps;
end

dims = ndims(Y);
if dims ~= 2
    error('Y has to be 2d in get_noise_fft_mod()');
end

sizY = size(Y);
N = min(sizY(end), options.max_timesteps);
% if N < sizY(end)
%     %Y = reshape(Y,prod(sizY(1:end-1)),[]);
%     switch ndims(Y)
%         case 2
%             Y(:,N+1:end) = [];
%         case 3
%             Y(:,:,N+1:end) = [];
%         case 4
%             Y(:,:,:,N+1:end) = [];
%     end
% end

Fs = 1;
ff = 0:Fs/N:Fs/2;
indf=ff>range_ff(1); % window in fourier domain
indf(ff>range_ff(2))=0;

d = prod(sizY(1:dims-1));
%Y = reshape(Y, d, N);
Nb = prod(block_size);
SN = cell(ceil(d/Nb),1); % this is just used to build a cell
PSDX = cell(ceil(d/Nb),1);
if ~split_data
    for ind = 1:ceil(d/Nb)
        xdft = fft(Y((ind-1)*Nb+1 : min(ind*Nb,d), 1:N), [], 2); % only fft in 2nd dimension, i.e. time domain
        xdft = xdft(:,1: floor(N/2)+1); % FN: floor added. this is because of symmetry
        psdx = (1/(Fs*N)) * abs(xdft).^2;
        psdx(:,2:end-1) = 2*psdx(:,2:end-1) + eps; % why do this thing?
        %SN{ind} = mean_psd(psdx(:,indf),method);
        switch lower(method)
            case 'mean'
                SN{ind}=sqrt(mean(psdx(:,indf)/2,2));
            case 'median'
                SN{ind}=sqrt(median(psdx(:,indf)/2,2));
            case 'logmexp'
                SN{ind} = sqrt(exp(mean(log(psdx(:,indf)/2),2)));
            otherwise
                error('unknown method for averaging noise..')
        end
        if nargout > 1
            PSDX{ind} = psdx;
        end
    end
else
    nc = ceil(d/Nb);
    Yc = mat2cell(Y(:, 1:N), [Nb * ones(nc-1,1); d - (nc-1) * Nb], N);
    parfor ind = 1:ceil(d/Nb)
        xdft = fft(Yc{ind},[],2);
        xdft = xdft(:,1:floor(N/2)+1);
        psdx = (1/(Fs*N)) * abs(xdft).^2;
        psdx(:,2:end-1) = 2*psdx(:,2:end-1) + eps;
        Yc{ind} = [];
        switch lower(method)
            case 'mean'
                SN{ind}=sqrt(mean(psdx(:,indf)/2,2));
            case 'median'
                SN{ind}=sqrt(median(psdx(:,indf)/2,2));
            case 'logmexp'
                SN{ind} = sqrt(exp(mean(log(psdx(:,indf)/2),2)));
            otherwise
                error('unknown method for averaging noise..')
        end
    end
end
sn = cell2mat(SN);
if nargout > 1
    PSDX = cell2mat(PSDX); % this is merge. interesting, save time
end
%disp([datestr(now, 'YYYY-mm-dd HH:MM:ss') ': Memory usage in get_noise_fft_mod(): ']); whos;
% if dims > 2
%     sn = reshape(sn,sizY(1:dims-1));
% end
end
