function [res] = im_shift(input,delx,dely)

[R,C] = size(input);
res = zeros(R,C,'single');
tras = single([1 0 delx; 0 1 dely;0 0 1]);

for i = 1:R
    for j = 1:C
        temp = [i;j;1];
        temp = tras*temp;
        x = temp(1,1);
        y = temp(2,1);
        if (x<=R)&&(y<=C)&&(x>=1)&&(y>=1)
            res(x,y) = input(i,j);
        end
    end
end
end
