function imgs_res = modcrop(imgs, modulo)
% first cut the boundary of the shifted image
% then crop the image to be an image with size correlated with modulo
if size(imgs,3)==1
    sz = size(imgs);
    sz = sz - mod(sz, modulo);
    imgs_res = imgs(1:sz(1), 1:sz(2));
else
    tmpsz = size(imgs);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    imgs_res = imgs(1:sz(1), 1:sz(2),:);
end
end

