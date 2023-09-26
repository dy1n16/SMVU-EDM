function [pidx, zidx] = get_positive_index(W)
% to get the positive indices of the weight matrix W
%    only take the upper part
% pidx -- positive indices of triu(W, 1);
% zdix -- zero indices of triu(W, 1) (only the upper part of W)

triuW = triu(W, 1);
pidx = triuW > 0;

triuW(pidx) = 0;
triuW(~pidx) = 1;
triuW = triu(triuW, 1);
zidx = triuW > 0;
end