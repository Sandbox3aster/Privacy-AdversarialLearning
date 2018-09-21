depth_imgs = subdir(fullfile('/home/wuzhenyu_sjtu/Desktop/noisy_version', 'depth*.png'))
for i = 1:size(depth_imgs,1)
    img_path = depth_imgs(i).name
    I = imread(img_path);
    BW = imbinarize(I);
    name = strrep(img_path,'depth','bin')
    imwrite(BW, name)
end
    
    