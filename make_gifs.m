%%

clear all
close all
clc

% %% 
% %-----------------------------------------------
% %  Vertical Contour Plot of Density (Horizontally sliced)
% %-----------------------------------------------
% % PC_num = 1;
% for PC_num = 1:5
%     
% data = importdata(sprintf('reconstructedmap%d.mat', PC_num)) ;
% 
%     lat = [ -87.5 : 5: 87.5] ;  
%     lon = [-180 : 5: 175] ;
% 
%     %%
% 
%     [x,y] = meshgrid( lon ,   lat);
%          fig = figure;
%          pos = get(fig,'position');
% 
% 
%     for t = [2:48]      
%           frame = getframe(fig); 
%            im = frame2im(frame); 
%            [imind,cm] = rgb2ind(im,256);
%                 [C,h] = contourf(x, y, data(:,:,t), 50);
%                 set(h,'LineColor','none')
%                 c = colorbar;
%     %           caxis([-40, -10])
%                 c.Label.String = '[kg/m^3]';
%                 xlabel('Longitude')
%                 ylabel('Latitude')
%                 title(['PC',num2str(PC_num),' Density Variations, TimeIndex = ', num2str(t)])
%     %             pause(0.000001)
%     %%%%% Write to the GIF File 
%           if t == 2 
%               imwrite(imind,cm,sprintf('reconstruction_%d.gif',PC_num),'gif', 'Loopcount',inf); 
%           else 
%               imwrite(imind,cm,sprintf('reconstruction_%d.gif',PC_num),'gif','DelayTime',0.1,'WriteMode','append'); 
%           end 
% 
% %      for tt=1:length(t)
% %      im=frame2im(frame(tt)); 
% %          [imind,cm] = rgb2ind(im,256);
% %          imwrite(imind,cm,sprintf('TESTreconstruction_%d.gif',PC_num), 'gif','DelayTime',0.1,'WriteMode','append');
% %     end
%     end
%     
% end
%% Make giff for all PCs

clear all
close all
clc
    
data = importdata('reconstructedmap_all.mat') ;

    lat = [ -87.5 : 5: 87.5] ;  
    lon = [-180 : 5: 175] ;

    [x,y] = meshgrid( lon ,   lat);
         fig = figure;
         pos = get(fig,'position');


    for t = [2:48]      
          frame = getframe(fig); 
           im = frame2im(frame); 
           [imind,cm] = rgb2ind(im,256);
                [C,h] = contourf(x, y, data(:,:,t), 50);
                set(h,'LineColor','none')
                c = colorbar;
    %           caxis([-40, -10])
                c.Label.String = '[kg/m^3]';
                xlabel('Longitude')
                ylabel('Latitude')
                title([' All PCs Density Variations, TimeIndex = ', num2str(t)])
    %             pause(0.000001)
    %%%% Write to the GIF File 
          if t == 2 
              imwrite(imind,cm,('reconstruction_all.gif'),'gif', 'Loopcount',inf); 
          else 
              imwrite(imind,cm,('reconstruction_all.gif'),'gif','DelayTime',0.1,'WriteMode','append'); 
          end

%          for tt=2:length(t)
%          im=frame2im(frame(tt)); 
%          [imind,cm] = rgb2ind(im,256);
%          imwrite(imind,cm,('TESTreconstruction_all.gif'), 'gif','DelayTime',0.1,'WriteMode','append');
%          end
    end
    


