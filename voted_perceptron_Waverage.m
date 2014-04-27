% Voted Perceptron 
% Author: Amir Azarbakht and Mandana Hamidi  <azarbaam@eecs.oregonstate.edu>
% Date: 2014-04-17
clc;
clear all;
close all;

% import the data
data = load('iris-twoclass.csv');
[dim1, dim2] = size(data);
% extract first column as class labels
Y = data(:,1);
% add dummy feature for w0, and extract the rest of the column as X
X = [ones(dim1,1), data(:,2:end);];

clear dim1 dim2;
[dim1, dim2] = size(X);

% store a collection of N linear separators
MAX_N = 2000;
w = zeros(MAX_N,dim2);
% store their survival time 
c = zeros(MAX_N+1,1);
% predicted output
u = ones(dim1,1);

err = zeros(dim1,1);

MAX_epoch = 500;
epoch_count = 0;
error_count = 0;
epoch_error = zeros(MAX_epoch,2);

n = 1;

% repeat 
for index = 1:MAX_epoch,
    epoch_count = epoch_count + 1;
    error_count = 0;    
    
    % learning the models w
    for i = 1:dim1,

        u(i,1) = w(n,:) * X(i,:)';
        
        if Y(i,1)*u(i,1) <= 0
           l = n + 1;
           w(l,:) = w(n,:) + (Y(i,1)*X(i,:));
           c(l,1) = 1;
           n = l;


%            error_count = error_count + 1;
        else
            c(n,1) = c(n,1) + 1;
        end
    end
    
    
    
    % prediction
    
    MAX_N = n;
    for i = 1:dim1,
    
        
    %   average perceptron, instead of sign of sum of sign of ...
    Wavg = zeros(1,dim2);
    for k = 1 : MAX_N,
        Wavg = Wavg + (c(k,1).*(w(k,:)'))';
    end
    
     if sign(Wavg(1,:)*X(i,:)') == 1
           u(i,1) = 1;
      else
            u(i,1) = -1;
     end
     
     
     %
        % sign of w * X
%         s = 0;
%         for j = 1:MAX_N,
%             if sign(w(j,:)*X(i,:)') == 1
%                 signwx = 1;
%             else
%                 signwx = -1;
%             end
%             sum over all weighted perceptrons
%             s = s + (c(j,1) * (signwx));
%         end
%       
%      s=signwx
%         if sign(s) == 1
%             u(i,1) = 1;
%         else
%             u(i,1) = -1;
%         end
        
        if Y(i,1)*u(i,1) <= 0
            error_count = error_count + 1;
            err(i,1) = 1;
        end
    end

    epoch_error(epoch_count, 1) = epoch_count;
    epoch_error(epoch_count, 2) = error_count;
 
end

% G = [u, X(:,2:end), err];
% 
% for i = 1:150
%     if G(i,4) == 0
%         G(i,:) =0;
%     end
% end
% 
% j = 1;
% for i = 1:150
%     
%     if G(i,4) ~= 0
%         G_final(j, :) = G(i, :);
%         j = j + 1;
%     end
% end
% 
% G_final


% W
% epoch_count
% epoch_error
% csvwrite('voted_perceptron.csv',W);

% scatter plot of data points
% scatter(getcolumn(data(:,2:end),1),getcolumn(data(:,2:end),2));figure(gcf)
% hold on;
% scatter(getcolumn(G_final(:,2:3),1),getcolumn(G_final(:,2:3),2));figure(gcf)
% hold on;
% scatter(getcolumn(data(1:100,2:end),1),getcolumn(data(1:100,2:end),2));figure(gcf)
% hold on;
% title('Voted Perceptron');

% plot of classification error as a function of the number of training
% epoches
% figure1 = figure('Color',[1 1 1]);

figure1 = figure('Color',[1 1 1]);
figure(1);
plot(epoch_error(1:end,end),'r');
figure(gcf);
title('Voted Perceptron: Classification Error as a function of the number of training epoches');
xlabel('Number of training epoches');
ylabel('Classification Error');

saveas(1, 'Voted_Perceptron_Wavg', 'png');
saveas(1, 'Voted_Perceptron_Wavg', 'epsc2');
saveas(1, 'Voted_Perceptron_Wavg', 'fig');
