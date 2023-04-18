function [neuron_trace_mat, deconvol_neuron_trace_mat, ...
          spike_mat, coefs_array] = component_refinement_module(T, T_bg, SI)
%% module for refining the neuron compoennt
%  Input
%       T: cell array, record the raw activities of each neuron
%       T_bg: cell array, record the raw activities of each neuropil
%       SI: SI: global config struct 
%  Output
%      neuron_trace_mat: raw activitis of neuronal activities
%       deconvol_neuron_trace_mat: deconvolved neuronal activities
%       spike_mat: deconvolved spike events
%       coefs_array: optimzied neuropil coefficients
%  last update: 5/31/2020. YZ.

%% parser
oasis_g = SI.oasis_g;
lambda_l0 = SI.lambda_l0;
oasis_lambda = SI.oasis_lambda;
%% temporal refinement
if ~isempty(T_bg)
    for i = 1 : length(T)      
        center_trace = T{i};
        bg_trace = T_bg{i};
        [coefs, sub_trace, ca_out, sp_out] = neuropil_coefficient_estimation_greedy(center_trace, ...
            bg_trace, oasis_lambda, oasis_g, lambda_l0);       
        neuron_trace_mat(i, :) = sub_trace;
        deconvol_neuron_trace_mat(i, :)  = ca_out;
        spike_mat(i, :) = sp_out;  
        coefs_array(i) = coefs;
    end   
else
    for i = 1 : length(T)
        center_trace = T_raw{i};
        [ca_out, sp_out, ~, ~, ~] = foopsi_oasisAR1(center_trace, oasis_g, oasis_lambda, false,...
         true, 1, 100);
        neuron_trace_mat(i, :) = center_trace;
        deconvol_neuron_trace_mat(i, :) = ca_out;
        spike_mat(i, :) = sp_out;     
    end
    coefs_array = [];
end