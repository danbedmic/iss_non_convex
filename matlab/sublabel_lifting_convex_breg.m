function [] = sublabel_lifting_convex_breg(cost_volume, gamma, lmb, iter_max, isotropic, save_to, backend)

% Lifted Bregman iteration for stereo matching problem. 
% Based on cvpr2016/sublabel_lifting_convex, extended to fit Bregman term
 

    % variables to store dimensions           
    [ny, nx, sublabels] = size(cost_volume);
    L = size(gamma, 1);                         % number of labels
    k = L - 1;                                  % dimension of lifted u
    N = ny*nx;

    % Bregman initialisation
    uk        = cell(1,iter_max);            % solution each iteration
    plbm      = zeros(2*N*k,1);              % gradient    

    % compute convex conjugate of data term
    subgamma = linspace(gamma(1), gamma(end), sublabels)';
    [px, py, indices, counts] = ...
        compute_convex_conjugate(cost_volume, L, subgamma, gamma);
    
    % compute prefactors
    gamma_vec_start = repmat(gamma(1:end-1), [N, 1]);
    gamma_vec_end = repmat(gamma(2:end), [N, 1]);
    lmb_scaled = lmb * (gamma(2) - gamma(1));

    % specify solver options
    opts = prost.options(...
        'max_iters', 50000, ...
        'num_cback_calls', 25, ...
        'solve_dual', false, ...
        'tol_rel_primal', 1e-13, ...
        'tol_rel_dual', 1e-13, ...
        'tol_abs_primal', 1e-13, ...
        'tol_abs_dual', 1e-13);
    
    %% Bregman iteration
    for iter_now = 1:iter_max
        
        % primal variables
        u = prost.variable(N*k);
        s = prost.variable(N*k);

        % dual variables
        vz = prost.variable(2*N*k);
        q = prost.variable(N);
        p = prost.variable(2*N*k);
        breg = prost.variable(2*N*k); 

        v = prost.sub_variable(vz, N*k);
        z = prost.sub_variable(vz, N*k);

        % new for anisotropic TV
        p1 = prost.sub_variable(p,N*k);
        p2 = prost.sub_variable(p,N*k);
               
        % primal dual formulation
        problem = prost.min_max_problem( {u,s}, {vz, q, p, breg} );

        % data term
         problem.add_function(vz, prost.function.transform( @(idx, count) ...
                 prox_sum_ind_epi_polyhedral_1d(idx, count / 2, false, px, py, ...
                         gamma_vec_start, gamma_vec_end, double(indices), double(counts)), ...
                 1 / (gamma(2) - gamma(1)), 0, 1, 0, 0));       
        
        % data term with lagrange multipliers                                              
        problem.add_function(q, prost.function.sum_1d('zero', 1, 0, 1, 1, 0));
        problem.add_dual_pair(u, v, prost.block.identity());
        problem.add_dual_pair(s, v, @(row, col, nrows, ncols) ...
                 block_dataterm_sublabel(row, col, nx, ny, L, gamma(1), gamma(end)));
        problem.add_dual_pair(s, z, prost.block.identity());
        problem.add_dual_pair(s, q, prost.block.sparse(kron(speye(N), (gamma(1:end-1) - gamma(2:end))')));
        
        % total variation regularizer
        problem.add_dual_pair(u, p, prost.block.gradient2d(nx, ny, k, true));
        
        if isotropic
            problem.add_function(p, ...
                prost.function.sum_norm2(2, false, 'ind_leq0', 1/lmb_scaled, 1, 1, 0, 0));
        else
            problem.add_function(p1, ...
                prost.function.sum_norm2(1, false, 'ind_leq0', 1/lmb_scaled, 1, 1, 0, 0));
            problem.add_function(p2, ...
                prost.function.sum_norm2(1, false, 'ind_leq0', 1/lmb_scaled, 1, 1, 0, 0));
        end

        % terms for Bregman iteration
        problem.add_function(breg, prost.function.sum_1d('ind_eq0', 1, -plbm, 1)); 
        problem.add_dual_pair(u, breg, prost.block.gradient2d(nx, ny, k, true)); 
    
        % solve problem
        prost.solve(problem, backend, opts);
        plbm = problem.dual_vars{3}.val;

        % u_volume is lifted solution. Get solution u_k in original space 
        % with the help of the layer-cake formula
        
        uk{iter_now} = (gamma(2:end)-gamma(1:end-1))' * reshape(u.val, [k, N]);
        uk{iter_now} = reshape(uk{iter_now}, ny, nx);
        uk{iter_now} = min(max(uk{iter_now}, 0), 1);
              
        save(save_to, 'uk'); 
        
        imshow(uk{iter_now})
        colormap(jet(256))
        title(iter_now)
        
    end


end
