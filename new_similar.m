function [S]=new_similar(rebuild_Label,origin_Label)

    L = logical(origin_Label); 
    F0 = rebuild_Label;
    [F_c, F_n] = size(F0);

      
    F_colsum = sum(F0, 1);
    F_colsum(F_colsum == 0) = 1;
    F_mid = F0 ./ F_colsum;  

        
    F = F_mid;



S = zeros(F_n, F_n);
F_nonzero = F ~= 0;  % F_c × F_n


parfor j = 1:F_n
    for z = 1:F_n
        mask_both = F_nonzero(:, j) & F_nonzero(:, z);
        F_j = F(mask_both, j);
        F_z = F(mask_both, z);
        
        S(j, z) = sum(2 - abs(F_j - F_z));
    end
end
end