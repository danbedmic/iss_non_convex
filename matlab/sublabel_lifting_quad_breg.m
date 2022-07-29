function [] = sublabel_lifting_quad_breg(im, gamma, lmb, backend, iter_max, special_gradient, save_to)

% Lifted Bregman iteration for ROF problem. 
% Based on cvpr2016/sublabel_lifting_quad, extended to fit Bregman term
    

    % variables to store dimensions
    [ny, nx, nc] = size(im);
    L            = size(gamma, 1);          % number of labels
    k            = L-1;                     % dimension of lifted u(x)
    N            = ny*nx;   
    
    % Bregman initialisation
    iter_now = 0;                           % maximal number of iterations
    uk       = cell(1,iter_max);            % solution each iteration
    plbm     = zeros(2*N*k,1);              % gradient    
      
    % compute piecewise quadratic approximation of dataterm
    polya = zeros(L - 1, ny, nx);
    polyb = zeros(L - 1, ny, nx);
    polyc = zeros(L - 1, ny, nx);
    
    for i=1:(L-1)
        polya(i,:, :) = 1;
        polyb(i,:, :) = -2*(im);
        polyc(i,:, :) = ((im).^2);
    end
    
    polya = polya(:);
    polyb = polyb(:);
    polyc = polyc(:);
    
    % compute prefactors
    gamma_vec_start = repmat(gamma(1:end-1), [N, 1]);
    gamma_vec_end   = repmat(gamma(2:end), [N, 1]);
    lmb_scaled      = lmb * (gamma(2) - gamma(1));


    % specify solver options
    opts = prost.options(...
        'max_iters', 70000, ...
        'num_cback_calls', 20, ...
        'tol_rel_primal', 1e-12, ...
        'tol_abs_primal', 1e-12, ...
        'tol_rel_dual', 1e-12, ...
        'tol_abs_dual', 1e-12);
    
    while iter_now < iter_max
        iter_now = iter_now + 1;
 
        % primal variables
        u = prost.variable(N*k);
        s = prost.variable(N*k);

        % dual variables including new dual variable for Bregman iteration
        vz = prost.variable(2*N*k);
        q = prost.variable(N);
        p = prost.variable(2*N*k);
        b = prost.variable(2*N*k);
        
        v = prost.sub_variable(vz, N*k);
        z = prost.sub_variable(vz, N*k);
        
        p1 = prost.sub_variable(p,N*k);
        p2 = prost.sub_variable(p,N*k);
        
        problem = prost.min_max_problem( {u,s}, {vz, q, p, b} );

        % data term
        problem.add_function(vz, ...
            prost.function.transform( @(idx, count) ...
                prox_sum_ind_epi_conjquad_1d(idx, count / 2, false, ...
                    polya, polyb, polyc, gamma_vec_start, gamma_vec_end), ...
                    1 / (gamma(2) - gamma(1)), 0, 1, 0, 0));

        % data term with lagrange multipliers                                           
        problem.add_function(q, prost.function.sum_1d('zero', 1, 0, 1, 1, 0));
        problem.add_dual_pair(u, v, prost.block.identity());
        problem.add_dual_pair(s, v, @(row, col, nrows, ncols) ...
             block_dataterm_sublabel(row, col, nx, ny, L, gamma(1), gamma(end)));
        problem.add_dual_pair(s, z, prost.block.identity());
        problem.add_dual_pair(s, q, ...
            prost.block.sparse(kron(speye(N), (gamma(1:end-1) - gamma(2:end))')));
        
        
        % total variation regularizer                                         
        problem.add_dual_pair(u, p, prost.block.gradient2d(nx, ny, k, true));
        
        problem.add_function(p1, ...
            prost.function.sum_norm2(1, false, 'ind_leq0', 1/lmb_scaled, 1, 1, 0, 0));
        problem.add_function(p2, ...
            prost.function.sum_norm2(1, false, 'ind_leq0', 1/lmb_scaled, 1, 1, 0, 0));
    
        % terms for Bregman iteration
        problem.add_function(b, prost.function.sum_1d('ind_eq0', 1, -plbm, 1)); 
        problem.add_dual_pair(u, b, prost.block.gradient2d(nx, ny, k, true)); 

        % solve problem
        prost.solve(problem, backend, opts);         
        plbm = problem.dual_vars{3}.val;       
        
        % u_volume is lifted solution. Get solution u_k in original space 
        % with the help of the layer-cake formula
        
        u_volume = reshape(u.val, [k, N]);
        uk{iter_now} = (gamma(2:end)-gamma(1:end-1))' * u_volume;

        % arbitary subgraient or subgradient of given shape?
        
        if special_gradient
                % reshape for easier calculation
                plbm_reshaped = reshape(plbm, [k,N,2]);

                % find index i of interes
                [~, ui] = max(repmat(uk{iter_now},k,1) <= gamma(2:end));
                ui(ui==0) = 1;

                % find special subgradient
                plbm_special = zeros(size(plbm_reshaped));
                
                for i=1:length(ui)
                    plbm_special(:,i,1) = plbm_reshaped(ui(i),i,1)/ (gamma(ui(i)+1) - gamma(ui(i))) * (gamma(2:end)-gamma(1:end-1));
                    plbm_special(:,i,2) = plbm_reshaped(ui(i),i,2)/ (gamma(ui(i)+1) - gamma(ui(i))) * (gamma(2:end)-gamma(1:end-1));
                end   

                plbm = plbm_special(:);  
                clear plbm_reshaped plbm_special;
        end
        
        uk{iter_now} = reshape(uk{iter_now}, ny, nx, nc);
        save(save_to, "uk")
        
    end   
end
