function dis = spike_distance(Z1, Z2)
%% function to estimate the distance between two arraies of spikes
    % to aviod misalignment, to a slight dilate
    window = ones(1, 20);
    Z1_exp = conv(Z1(:), window(:), 'same');
    Z2_exp = conv(Z2(:), window(:), 'same');
    dis = sum((Z1_exp - Z2_exp).^2);
end