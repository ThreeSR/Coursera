function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%改变y从1-10到矩阵
c = 1:num_labels;%num = 10
yt = zeros(m,num_labels); % m =5000 m is training examples. batch
for i = 1:m
    yt(i,:) = (c==y(i));% 5000*10矩阵 logic operation 
end


a1 = [ones(m, 1) X];    %5000x401 add bias
a2 = sigmoid(a1 * Theta1');  % forward prop

a2 = [ones(m, 1) a2];    %5000x26
hx = sigmoid(a2 * Theta2');  

% cost function 
J = 1 / m * sum(sum(-yt.*log(hx) - (1-yt).*log(1-hx)));

%reg
reg = lambda / 2 / m * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))); %theta0 will not be included in reg calculation

% cost + reg
J = J + reg;

% grad back prop
delta_3 = hx - yt;                                               
delta_2 = delta_3 * Theta2 .* (a2 .* (1-a2)); 
delta_2 = delta_2(:,2:end);   

% Accumulate the gradient 
D2 = delta_3' * a2;    
D1 = delta_2' * a1;    

Theta2_grad = 1/m * D2;%不同的梯度get到不同的theta

Theta1_grad = 1/m * D1;    
%add reg

temp1 = Theta1;
temp2 = Theta2;
temp1(:,1) = 0; % 去除首项
temp2(:,1) = 0; 

Theta1_grad = Theta1_grad + lambda/m * temp1;%anderew 此处公式于PPT上有误，应该是少了括号。
Theta2_grad = Theta2_grad + lambda/m * temp2;%此处式子是原公式vectorize之后的结果

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
