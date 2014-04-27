% q2 Voted Perceptron

% Voted Perceptron 
% Author: Amir Azarbakht and Mandana Hamidi  <azarbaam@eecs.oregonstate.edu>
% Date: 2014-04-17
clc;
clear all;
close all;

% import the data
data = load('iris-twoclass.csv');
[dim1, dim2] = size(data);
X1 = data(:,2);
X2 = data(:,3);

x1 = (min(X1): (max(X1)-min(X1))/100: max(X1));
x2 = (min(X2): (max(X2)-min(X2))/100: max(X2));

gridData = [];
for i = 1 : size(x1,2)
    for j = 1 : size(x2,2)
       gridData = [ gridData; 1,  x1(i) , x2(j) ];  
    end
end

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

MAX_epoch = 100;
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
        % sign of w * X
        s = 0;
        for j = 1:MAX_N,
            if sign(w(j,:)*X(i,:)') == 1
                signwx = 1;
            else
                signwx = -1;
            end
            % sum over all weighted perceptrons
            s = s + (c(j,1) * (signwx));
        end
        
        
        if sign(s) == 1
            u(i,1) = 1;
        else
            u(i,1) = -1;
        end
        
        if Y(i,1)*u(i,1) <= 0
            error_count = error_count + 1;
            err(i,1) = 1;
        end
    end

    epoch_error(epoch_count, 1) = epoch_count;
    epoch_error(epoch_count, 2) = error_count;

end


uGrid = zeros(size(gridData,1),1);
% predict the gridData
    for i = 1:size(gridData,1),
        % sign of w * X
        s = 0;
        for j = 1:MAX_N,
            if sign(w(j,:)*gridData(i,:)') == 1
                signwxGrid = 1;
            else
                signwxGrid = -1;
            end
            % sum over all weighted perceptrons
            s = s + (c(j,1) * (signwxGrid));
        end
        
        if sign(s) == 1
            uGrid(i,1) = 1;
        else
            uGrid(i,1) = -1;
        end
    end
    
    

    
    j = 0;
k = 0;
for i = 1:size(data,1),
    if data(i,1) > 0
        j = j + 1;
        posData(j,:) = data(i,2:end);
    else
        k = k + 1;
        negData(k,:) = data(i,2:end);
    end
end


    
posGridData = [];
negGridData = [];

for i = 1 : size(uGrid,1),
    if uGrid(i,1) > 0
        posGridData = [posGridData ;gridData(i,2:end)];
    else
        negGridData = [negGridData; gridData(i,2:end)];
    end
end


% scatter plot of the data, and the learned linear classifier
figure1 = figure('Color',[1 1 1]);
figure(1);
scatter(getcolumn(negData,1),getcolumn(negData,2),'r', 'o');
%figure(gcf)
hold on; 
scatter(getcolumn(posData,1),getcolumn(posData,2),'b', 'x');
%figure(gcf)
hold on;

scatter(getcolumn(negGridData,1),getcolumn(negGridData,2),'m', '+');
%figure(gcf)
hold on; 
scatter(getcolumn(posGridData,1),getcolumn(posGridData,2),'c', '+');
%figure(gcf)
legend('Positive class', 'Negative class');

%ezplot([num2str(w(1)) ' + ' num2str(w(2)) '* x ' num2str(w(3)) '* y = 0' ,]);
xlabel('x_1');
ylabel('x_2');
title('Voted Perceptron Decision Boundary Grid');

saveas(1, 'Voted_Perceptron_grid', 'png');
saveas(1, 'Voted_Perceptron_grid', 'epsc2');
saveas(1, 'Voted_Perceptron_grid', 'fig');
