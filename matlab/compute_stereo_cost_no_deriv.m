function [cost_volume] = compute_stereo_cost_no_deriv(path_im0, path_im1, ndisps, dscl1)
% computes a stereo matching cost between two images given by
% path_im0 and path_im1. ndisps denotes the number of disparities,
% dscl1 is a factor by which to downscale the original image.
    
    % read stereo images
    im_0 = imread(path_im0);
    im_1 = imread(path_im1);
    
    im_0 = imresize(double(im_0) / 255, (1/dscl1));
    im_1 = imresize(double(im_1) / 255, (1/dscl1));
    
    [ny, nx, ~] = size(im_0);

    % compute full dataterm
    cost_volume = zeros(ny, nx, ndisps);

    for d=1:ndisps
        cost_volume(:, :, d) = min(sum(abs([repmat(im_1(:, 1, :), 1, d-1) im_1(:, 1:(nx-d+1), :)] - im_0), 3),0.1);
    end
    

end
