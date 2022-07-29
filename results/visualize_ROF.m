
%% Fig 3

l2 = load('ROF_L2.mat');
l5 = load('ROF_L5.mat');
l5p = load('ROF_L5+.mat');

figure
for i=1:5
    sgtitle("Iteration k = "+num2str(i))

    subplot(1,3,1)
    surf(l2.uk{i}, 'EdgeColor', 'none', 'FaceColor', 'interp')
    caxis([0, 1])
    zlim([0, 1])
    colormap turbo
    axis off
    title('Bregman iteration')
    

    subplot(1,3,2)
    surf(l5.uk{i}, 'EdgeColor', 'none', 'FaceColor', 'interp')
    caxis([0, 1])
    zlim([0, 1])
    colormap turbo
    axis off
    title('Lifted Bregman iteration - arbitrary subgradient')

    subplot(1,3,3)
    surf(l5p.uk{i}, 'EdgeColor', 'none', 'FaceColor', 'interp')
    caxis([0, 1])
    zlim([0, 1])
    colormap turbo
    axis off
    title('Lifted Bregman iteration - transformed subgradient')
    colorbar()
    
    pause(2)
end