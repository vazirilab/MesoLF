function [ ] = stack_viewer(stack)
%PSF_VIEWER Show interactive plot of PSF
f = figure;
z = ceil(size(stack,3)/2);
ax = axes('Parent', f, 'position', [0.13 0.20 0.77 0.77]);
%scale_min = prctile(stack(:), 0.01);
scale_min = min(stack(:));
scale_max = prctile(stack(:), 99.99);
if scale_max == 0
    scale_max = max(stack(:));
end
imagesc(squeeze(stack(:,:,z)), 'Parent', ax, [scale_min scale_max]);
colorbar();
axis image;
bz = uicontrol('Parent', f, 'Style', 'slider', 'Position', [81,14,419,10], ...
    'value', z, 'min', 1, 'max', size(stack, 3), 'SliderStep', [1, 1] / (size(stack, 3) - 1));
set(bz, 'Callback', @(es, ed) update_z(es, ed));
%hLstnz = handle.listener(bz, 'ActionEvent', @update_z); %#ok<NASGU>

    function update_z(es, ed)
        z = floor(get(es,'Value'));
        update_plot();
    end

    function update_plot()
        disp(z);
        imagesc(squeeze(stack(:,:,z)), 'Parent', ax, [scale_min scale_max]);
        colorbar();
        axis image;
%         grid on; 
%         set(gca, 'XTick', 0.5:11:300);
%         set(gca, 'XColor', 'w');
%         set(gca, 'YTick', 0.5:11:300);
%         set(gca, 'YColor', 'w');
    end
end

