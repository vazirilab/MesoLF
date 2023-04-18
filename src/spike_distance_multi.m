function dis = spike_distance_multi(Z1, Z2)
%% function to estimate the distance between two arraies of spikes
%  this file could be used for pdist
%  Z1 is 1-by-n observation
%  Z2 is m2-by-n list of observations
%  last update: 2/14/2020

    % to aviod misalignment, to a slight dilate
    window = ones(1, 20);
    m2 = size(Z2, 1);
    dis = zeros(m2, 1);
    for i = 1 : m2
    Z1_exp = conv(Z1, window, 'same');
    Z2_exp = conv(Z2(i, :), window, 'same');
    dis(i) = sum((Z1_exp - Z2_exp).^2);
    end
end