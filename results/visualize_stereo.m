

%% Fig 5

iso = load('Stereo_Rainbow_Isotropic.mat');
aniso = load('Stereo_Rainbow_Anisotropic.mat');

figure
for i=1:10
    sgtitle("Iteration k = "+num2str(i))

    subplot(1,2,1)
    surf(iso.uk{i}, 'EdgeColor', 'none', 'FaceColor', 'interp')
    caxis([0, 1])
    zlim([0,1])
    colormap turbo
    axis off
    title('Isotropic exmaple')
    

    subplot(1,2,2)
    surf(aniso.uk{i}, 'EdgeColor', 'none', 'FaceColor', 'interp')
    caxis([0, 1])
    zlim([0,1])
    colormap turbo
    axis off    
    title('Anisotropic exmaple')
    colorbar()
    
    pause(2)
end