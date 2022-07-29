function [] = rof(data)

iter_max = 5;
save_to = "../results/ROF " + string(datetime) + ".mat";

img_noisy = double(imread("../images/circles_noisy_18.jpg"))/255;

switch data
    case 1
        L = 2;
        special_gradient = false;
        lmb = 20;

        backend = prost.backend.pdhg(...
            'stepsize', 'alg2', ...
            'alg2_gamma', 0.018,...
            'arb_delta', 1.05,...                
            'arb_tau', 0.8,... 
            'arg_alpha0', 0.5,... 
            'arg_delta', 1.5,... 
            'arg_nu', 0.95, ...             
            'residual_iter', 16, ...
            'scale_steps_operator', 1, ...
            'sigma0', 0.09, ...
            'tau0', 15);
        
    case 2
        L = 5;
        special_gradient = false;
        lmb = 20;

        backend = prost.backend.pdhg(...
            'stepsize', 'alg2', ...
            'alg2_gamma', 0.00018,...
            'arb_delta', 1.05,...                
            'arb_tau', 0.8,... 
            'arg_alpha0', 0.5,... 
            'arg_delta', 1.5,... 
            'arg_nu', 0.95, ...             
            'residual_iter', 1000, ...
            'scale_steps_operator', 1, ...
            'sigma0', 0.09, ...
            'tau0', 15);        

        
    case 3
        L = 5;
        special_gradient = true;
        lmb = 20;

        backend = prost.backend.pdhg(...
            'stepsize', 'alg2', ...
            'alg2_gamma', 0.00018,...
            'arb_delta', 1.05,...                
            'arb_tau', 0.8,... 
            'arg_alpha0', 0.5,... 
            'arg_delta', 1.5,... 
            'arg_nu', 0.95, ...             
            'residual_iter', 1000, ...
            'scale_steps_operator', 1, ...
            'sigma0', 0.09, ...
            'tau0', 15);
end

% solve problem
sublabel_lifting_quad_breg(img_noisy, linspace(0, 1, L)', lmb, backend, iter_max, special_gradient, save_to);


% visualize
uk = load(save_to, 'uk');
figure
hold off

for i = 1:iter_max
    surf(uk.uk{i}, 'EdgeColor', 'none', 'FaceColor', 'interp')
    caxis([0, 1])
    colormap turbo
    axis off
    title("Iteration k = "+num2str(i))
    pause(2)
end