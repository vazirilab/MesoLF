function [S_bg_mask_init, S_bg_init] = library_refinement(valid_seg, bg_spatial_init, ...
    S_bg_mask_init, S_bg_init, S_mask_init, bias)

%% function to exclude neurons from neropil segments
%  last update: 6/25/2020. YZ
disp('Refining library')

for i = 1 : size(valid_seg, 1)
    buf = valid_seg{i, 2};
    neuron_center(i, :) = mean(buf, 1);
end
% get patch size
patch_size = size( valid_seg{1, 1}, 1);   
neighbour_radius = 7 * patch_size;
% for each component

figure(401),
for i = 1 : length(S_bg_init)
    % find all neurons in a neighbour
    curr_center = neuron_center(i, :);
    
    % find
    to_curr_center_dist = sqrt(sum((neuron_center - curr_center).^2, 2)); % an array
    neighbor_neuron_ind = find((to_curr_center_dist<neighbour_radius) .* (to_curr_center_dist > 0));

    if ~isempty(neighbor_neuron_ind)
        curr_mask_pos = bias{i};
        curr_mask = zeros(size(bg_spatial_init));
        curr_mask(curr_mask_pos(1, 1) : curr_mask_pos(2, 1), ...
            curr_mask_pos(1, 2) : curr_mask_pos(2, 2)) = S_bg_mask_init{i};
        % exclude that part
        %subplot(1, 2, 1), imagesc(S_bg_mask_init{i}), axis equal, axis off, title(sprintf('mask %d, before refinement', i))
        for j = 1 : length(neighbor_neuron_ind)
            % restore global mask
            target_mask_pos = bias{neighbor_neuron_ind(j)};
            target_mask = zeros(size(bg_spatial_init));
            target_mask(target_mask_pos(1, 1) : target_mask_pos(2, 1), ...
                target_mask_pos(1, 2) : target_mask_pos(2, 2)) = ...
                S_mask_init{neighbor_neuron_ind(j)};

            % exclude overlapped part
            curr_mask = curr_mask .* (~target_mask);
        end
        S_bg_mask_init{i} = curr_mask(curr_mask_pos(1, 1) : curr_mask_pos(2, 1), ...
            curr_mask_pos(1, 2) : curr_mask_pos(2, 2));

        S_bg_init{i} = S_bg_init{i} .* S_bg_mask_init{i};

        %subplot(1, 2, 2), imagesc(S_bg_mask_init{i}), ...
        %axis equal, axis off, title(sprintf('mask %d, after refinement', i))
%         pause(1)
    end
end
end