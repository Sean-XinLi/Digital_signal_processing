%% 
% *Question 1*
% 
% Remember to assign your answers to the pre-existing variables, and do not 
% change their names. You may create other variables if you wish, but the auto-grader 
% will not see them.

% YOUR CODE BELOW
h1 = zeros(11, 1); %REPLACE WITH YOUR ANSWER

h2 = zeros(11, 1); %REPLACE WITH YOUR ANSWER

h3 = zeros(11, 1); %REPLACE WITH YOUR ANSWER

% PLOTTING - NO NEED TO EDIT
disp('Question 1 Figures')
figure; 
subplot(2,2,1); stem(-5:5, h1); title('h_1');
xlabel('Samples'); ylabel('Amplitude');
subplot(2,2,2); stem(-5:5, h2); title('h_2');
xlabel('Samples'); ylabel('Amplitude');
subplot(2,2,3); stem(-5:5, h3); title('h_3');
xlabel('Samples'); ylabel('Amplitude');
%% 
% *Question 2*

x = zeros(11, 1); %REPLACE WITH YOUR INPUT SIGNAL

n = 0; %REPLACE WITH YOUR ANSWER

h4 = 0; %REPLACE WITH IMPULSE RESPONSE OF SYSTEM
y4 = 0; %REPLACE WITH OUTPUT OF SYSTEM

h5 = 0; %REPLACE WITH IMPULSE RESPONSE OF SYSTEM
y5 = 0; %REPLACE WITH OUTPUT OF SYSTEM

h6 = 0; %REPLACE WITH IMPULSE RESPONSE OF SYSTEM
y6 = 0; %REPLACE WITH OUTPUT OF SYSTEM

h7 = 0; %REPLACE WITH IMPULSE RESPONSE OF SYSTEM
y7 = 0; %REPLACE WITH OUTPUT OF SYSTEM

% PLOTTING - NO NEED TO EDIT
disp('Question 2 Figures')
figure; 
set(gcf, 'Units', 'Inches', 'Position',[0 0 2.5 1.2])
stem(-5:5, x); title('Input Signal')
xlabel('Samples'); ylabel('Amplitude');

figure; 
subplot(2,2,1); stem(n, y4); title('y_4');
xlabel('Samples'); ylabel('Amplitude');
subplot(2,2,2); stem(n, y5); title('y_5');
xlabel('Samples'); ylabel('Amplitude');
subplot(2,2,3); stem(n, y6); title('y_6');
xlabel('Samples'); ylabel('Amplitude');
subplot(2,2,4); stem(n, y7); title('y_7');
xlabel('Samples'); ylabel('Amplitude');

%% 
% *Question 3*

%LOAD THE PROVIDED MAT FILE (PLEASE DON'T REMOVE)
load('code2_fall21.mat')

%YOUR CODE HERE

%Feel free to experiment!
%You can create your own plots and zoom in to check your answer.

%ANSWER
location = nan; %REPLACE WITH YOUR ANSWER

%% 
% *Functions*
% 
% Here's some space for function definitions, if you choose to make any.