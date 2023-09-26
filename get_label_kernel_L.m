function [L] = get_label_kernel_L(y)
% y is the label information of each point
% if y_i \not= y_j, L(i,j) = -1
% diagoanl: L(i,i) = total number of -1 on each row
%
n = length(y);
L = zeros(n,n);
for i = 1:n-1
    for j = (i+1):n
        if y(i) ~= y(j)
            L(i,j) = -1;
        end
    end
end
L = L + L';
s = sum(L, 2);
L(1:(n+1):end) = -s;
end
