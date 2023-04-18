function [ ret ] = is_positive_integer_or_zero( x )
if ~isnumeric(x)
    ret = false;
else
    ret = all(x == inf | (x >= 0 & ~mod(x, 1)));
end