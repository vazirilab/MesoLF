function pdist_array = seed_pdist(S, bias, movie_size)
%% this file is used to calculate the pairwise similarity of two patches
%  last update: 9/18/2020. YZ

%%
ind = 1;
pdist_array = zeros(length(S) - 1, length(S));
for i = 1 : length(S) - 1
    i
    curr_i = S{i};
    buf_i = zeros(movie_size(1), movie_size(2));
    buf_i(bias{i}(1, 1) : bias{i}(2, 1), bias{i}(1, 2) : bias{i}(2, 2)) = curr_i;
    buf_i = buf_i(:) / sqrt(sum(buf_i(:).^2));
   for j = i + 1 : length(S)
        curr_j = S{j};
        buf_j = zeros(movie_size(1), movie_size(2));
        buf_j(bias{j}(1, 1) : bias{j}(2, 1), bias{j}(1, 2) : bias{j}(2, 2)) = curr_j;
%         buf= corrcoef(buf_i(:), buf_j(:));
        buf_j = buf_j(:) / sqrt(sum(buf_j(:).^2));
        buf = buf_i.' * buf_j;
        pdist_array(i, j) = buf;     
   end    
end
pdist_array = get_tri_element(pdist_array);

end

% get triu elements
function M = get_tri_element(A)
M = A(triu(true(size(A)),1));
end