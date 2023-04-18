function  [config_out, config_valid] = mesolf_set_params(varargin)
if isunix()
    psf_cache_dir_default = '/dev/shm';
else
    psf_cache_dir_default = tempdir();
end

config_definition = {
    % first column is param name
    % second column is validator function handle
    % third column is default value; string 'required' to indicate required value
    % fourth column is parser function handle, which converts a string value (from command line or key-value argument) to required data type. 
    %   If empty and default is numeric, str2num() is used. Otherwise, no parsing is done.

    
%%% REQUIRED PARAMS

    % Input directory containing raw LFM frames in single-page tiff files
    {'indir',                       @isfolder                              'required'                       @strip};
    
    % Directory to place output files in
    {'outdir',                      @is_legal_pathname                     'required'                       @strip};
    
    % LFM PSF file as generated using LFrecon package 
    % (https://media.nature.com/original/nature-assets/nmeth/journal/v11/n7/extref/nmeth.2964-S2.zip)
    {'psffile',                     @is_existing_file                      'required'                       @strip};
    
    % Horizontal position of center of central microlens in LFM raw frames
    % (determine using http://graphics.stanford.edu/software/LFDisplay/)
    {'frames_x_offset',             @isfloat                               'required'                       @str2num};
    
    % Vertical position of center of central microlens in LFM raw frames
    % (determine using http://graphics.stanford.edu/software/LFDisplay/)
    {'frames_y_offset',             @isfloat                               'required'                       @str2num};
    
    % Microlens pitch in LFM raw frames
    % (determine using http://graphics.stanford.edu/software/LFDisplay/)
    {'frames_dx',                   @isfloat                               'required'                       @str2num};
    

    
%%% OPTIONAL PARAMS
    
 %%% Frames to include
    
    % Frame indices start:step:end to load.
    {'frames_start',                @is_positive_integer_or_zero,          1};
    {'frames_step',                 @is_positive_integer_or_zero,          1};
    {'frames_end',                  @is_positive_integer_or_zero,          inf};
    
    % Whether to average over the frames_step frames in between two steps
    % of the list of rames defined by frames_start : frames_step : frames_end
    {'frames_average',              @islogical,                            true};
    
    % Imaging frame rate [Hz]. Only used for axis ticks
    {'frameRate',                   @is_positive_integer_or_zero,          10};

    % Directory with very high access speed for PSF file caching
    % (ideally, use a RAM-disk such as /dev/shm)
    {'psf_cache_dir',               @is_in_existing_dir,                   psf_cache_dir_default,           @strip};

    % List of GPU IDs to use throughout the pipeline. default: [].  
    % (Note that in Matlab, gpu_ids start with 1, not 0, as in the output of nvidia-smi)
    {'gpu_ids',                     @(x) isnumeric(x) && all(floor(x) == x),      []};

    % Number of microlenses to crop from input frames on each side 
    % [left right top bottom] (as displayed in Fiji), to avoid border artefacts
    % When giving a value of 
    % floor([ix1_lo_border_width ix1_hi_border_width ix2_hi_border_width ix2_hi_border_width] / Nnum)
    % that means that
    % cropped_img = full_img(ix1_lo_border_width + 1 : end - ix1_hi_border_width, ix2_lo_border_width + 1 : end - ix2_hi_border_width)
    {'frames_crop_border_microlenses',     @(x) isnumeric(x) && all(size(x) == [1 4]) && all(floor(x) == x),      [0 0 0 0]};
    
    % Path to a .tif image containinag a mask for the valid FOV, or 'true' to use default, or 'false' to disable mask.
    % Neuron candidate segments with centers outside of valid mask are will be discarded. 
    % Set mask to 0 outside of valid area, any other value elsewhere.
    {'mask_file',                   @(x) islogical(x) || is_existing_file(x),  false};

 %%% Patch mode
    % If set to 0, then this process will sequentially work on all patches 
    % and perform the final merge. If set to an integer between 1 and the total number of patches, 
    % then this process will only work on the corresponding patch. If set
    % to a value larger than the number of patches, then this process will
    % perform the final merge and postprocessings.
    {'worker_ix',                   @is_positive_integer_or_zero,          0};
    
    % 'carpet' (tile into n_patches(1) by n_patches(2) patches) or 'roi' (tiles are set to ROIs specified in ROI_list)
    {'patch_mode',                  @ischar,                               'carpet',                        @strip};

    % Number of tiles per side in carpet mode, can be a vector [patch_rows
    % patch_cols]
    {'n_patches',                      @is_positive_integer_or_zero,          1};

    % List of ROIs to be used in ROI mode, given as list of ROI position vectors: 
    % [[roi0_i0 roi0_j0 roi0_i1 roi0_j1], [roi1_i0 roi1_j0 roi1_i1 roi1_j1], ...]
    % (Coordinates, assuming an image array in matlab, as returned by imread(): We name 1st index: i, 2nd index: j. 
    % Note that this is transposed compared to the x (left to right) and y
    % (top to bottom) coordinates in ImageJ: i = y, j = x.)
    {'ROI_list',                    @(x) isnumeric(x) && all(floor(x(:)) == x(:)),      []};

    % In ROI mode, patch size (in units of rectified pixels) to subdivide each ROI into. Should be multiple
    % of rectified microlens size (will be corrected to nearest multiple if
    % it is not).
    {'ROI_patch_size',              @isnumeric,                            []};

 %%% Registration (motion correction)
    % boolean; whether perform registration (motion correction)
    {'reg_flag',                    @islogical,                            false};
    % event size in registration
    {'reg_bin_width',               @is_positive_integer_or_zero,          20};

 %%% Detrending
    % 'pixelwise' or 'global'
    {'detrend_mode',                @ischar,                               'pixelwise',                     @strip};

    % window size for 'global' detrend mode
    {'global_detrend_delta',        @is_positive_integer_or_zero,          1000};

    % Number of iterations to perform for background low-rank factorisation
    {'bg_iter',                     @is_positive_integer_or_zero,          5};

    % block size along time
    {'block_size',                  @is_positive_integer_or_zero,          5};

    % large moving window to reduce motion and blood vessel artifact, only applies to 'pixelwise' detrend mode
    {'pixelwise_window_1',          @is_positive_integer_or_zero,          100};

    % small moving window to reduce noise, applies only to 'pixelwise' detrend mode
    {'pixelwise_window_2',          @is_positive_integer_or_zero,          20};

    % power index to enhance image, applies only to 'pixelwise' detrend mode
    {'pixelwise_poly_index',        @is_positive_integer_or_zero,          3};

 %%% 3D reconstruction
    % No. of PSF z planes (symmetric around central plane, value has to be odd) 
    % to use for finding neurons (z planes further out will be discarded after 
    % reconstruction to avoid border artefacts)
    {'valid_recon_range',           @(x) isinteger(x) && @(x) mod(x,2) == 1,      51};
    
    % 'phase_space' or 'phase_space_peeling'
    {'recon_mode',                  @ischar,                               'phase_space',           @strip};

    % Max. number of iterations to perform for reconstruction
    {'recon_max_iter',              @is_positive_integer_or_zero,          3};

    % Boolean; whether to mask out blood vessels before segmentation
    {'vessel_mask_enable',          @islogical,                            false};

    % Threshold for segmenting blood vessels
    {'vessel_thresh',               @isfloat,                              0.25};

 %%% Segmentation
    % maximum neuron number found in each depth
    {'neuron_number',               @is_positive_integer_or_zero,          150};

    % neuron lateral size
    {'neuron_lateral_size',         @is_positive_integer_or_zero,          15};

    % local contrast threshold to pick up neurons, usually 1~2
    {'local_constrast_th',          @isfloat,                              1.3};

    % Ratio of axial to lateral psf size
    {'optical_psf_ratio',           @isfloat,                              3};

    % threshold for merging neurons, usually 0.5~1
    {'overlap_ratio',               @isfloat,                              0.5};

    % Line width for boxes around segments in segmentation result plot
    {'boundary_thickness',          @is_positive_integer_or_zero,          3};

    % Whether to save discarded components
    {'discard_component_save',      @islogical,                            false};
   
	% Default segmentation methods, can be 'morph' or 'NMF' or 'mix'. 'morph' is
	% highly recommended
    {'segmentation_method',         @ischar,                               'morph'};
    
    % NMF segmentation: binarization threshold
    {'NMF_threshold',               @isfloat,                              0.1};
    
    % NMF segmentation: iteration
    {'NMF_iteration',               @isfloat,                              5};    
    
 %%% Library generation
    % How to store library of LFM footprints in memory: 'snippet' (just a small patch plus offset coordinates) or 'full' (full frames)
    {'lib_store_mode',              @ischar,                               'snippet',                         @strip};

    % If lib_store_mode is 'full', then this is ignored (results in the equivalent of 'ballistic').
    % If lib_store_mode is 'snippet', then this can be 'bg_ring' or 'ballistic'
    {'lib_gen_mode',                @ischar,                               'bg_ring',                       @strip};

    % Ratio of typical neuropil size to neuron size
    {'radius_ratio',                @isfloat,                              2};

 %%% NMF
    % Number of iterations for NMF
    {'nmf_max_iter',                @is_positive_integer_or_zero,          20};

    % Number of main iterations for NMF
    {'nmf_max_round',               @is_positive_integer_or_zero,          5};

 %%% Refinement of temporal signals
    % g parameter for oasis
    {'oasis_g',                     @isfloat,                              0.9};

    % regularization for oasis
    {'oasis_lambda',                @isfloat,                              0.02};

    % l0 reguarliztion parameter for background subtraction
    {'lambda_l0',                   @isfloat,                              0.01};

 %%% Merging
    % spatial threshold for merging
    {'spatial_merging_threshold',   @isfloat,                              30};

    % note: 1 - spatial_simi_threhold will be used for distance
    {'spatial_simi_threhold',      @isfloat,                               0.2};
    
    % note: 1 - temporal_merging_threshold will be used for distance
    {'temporal_merging_threshold',  @isfloat,                              0.5};

    % distance threshold for merging
    {'margin_dis_threshold',        @isfloat,                              15};

 %%% Quality check of temporal signals
    % classifier method, can be 'svm' or 'cnn'
    {'filt_method',                 @ischar,                               'svm',                           @strip};
    % classification mode, can be 'sensitive' or 'conservative'
    {'trace_keep_mode',             @ischar,                               'sensitive',                     @strip};
    
 %%% Summary plots
    % Enable summary plots of results (saved as pdf) at end of MesoLF run
    {'summary_plots_enable',         @islogical,                           true};   

};

%% Check varargin and assemble struct from key-value pars if necessary
if nargin == 1 
    if isa(varargin{1}{1}, 'struct')
        config_in = varargin{1}{1};
    else
        error('If only a one argument is given, it has to be a struct, with fields according to the required and optional parameters (see docs).');
    end
elseif nargin < 1 || mod(nargin, 2)
    error('Expecting either a single struct or a sequence of key-value pairs as input arguments.')
else
    % If the input argument is not a struct, then we expect a variable number of key-value pairs. 
    % The keys have to be strings. The values may be strings that can be parsed to the required data type
    config_in = struct;
    for i = 1:2:nargin
        key = varargin{i};
        val = varargin{i + 1};
        config_in.(key) = val;
    end
end

%% Loop over fields in config_definition. Check if it exists in config_in. If yes, use that value. Else, use default
config_out = struct;
config_valid = true;
for i = 1:size(config_definition, 1)
    field_name = config_definition{i}{1};
    validator = config_definition{i}{2};
    default = config_definition{i}{3};
    if numel(config_definition{i}) > 3
        parser = config_definition{i}{4};
    elseif isnumeric(default) || islogical(default)
        parser = @str2num;
    else
        parser = @(x) x;  % no operation
    end
    
    try
        if isfield(config_in, field_name)
            val = config_in.(field_name);            
            if validator(val)
                config_out.(field_name) = val;
            elseif validator(parser(val))
                config_out.(field_name) = parser(val);
            else
                error(['Parse/validation error for key ' field_name '. Invalid value was : ' val]);
            end
        elseif ~strcmp(default, 'required')
            config_out.(field_name) = default;
        else
            config_valid = false;
            disp(['Required argument not given:' field_name]);
        end
    catch my_error
        disp(['Error while processing input: key=' field_name ' val=']);
        disp(val);
        rethrow(my_error);        
    end
end
