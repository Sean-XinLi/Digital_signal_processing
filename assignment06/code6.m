clear;
close all;

%% 
% 
% *Question 1*: 

%% SETUP - DO NOT EDIT

% DEFINE VARIABLES
dt = 0.25;
n = -10:10;
tnew = -10:dt:10;
w = -4*pi:pi/20:(4*pi-pi/20);
M = length(w);
N = length(n);

 
%% ANSWER

% DEFINE YOUR INPUTS
ii = 1;
t{ii} = -10:2:8; %INSERT ANSWER HERE%  sampled time array
x{ii} = [0,1,0,-1,0,1,0,-1,0,1]; %INSERT ANSWER HERE%  sampled signal
H{ii} = (w>=-0.5*pi) - (w>=0.5*pi); %INSERT ANSWER HERE%  ideal filter (in frequency)
    
ii = ii + 1;
t{ii} = -10:1:9; %INSERT ANSWER HERE%  sampled time array
x{ii} = cos((3*pi)/4.*t{ii}); %INSERT ANSWER HERE%  sampled signal
H{ii} = (w>=-1*pi) - (w>=1*pi); %INSERT ANSWER HERE%  ideal filter (in frequency)

ii = ii + 1;
t{ii} = -10:0.25:9.75; %INSERT ANSWER HERE%  sampled time array
x{ii} = cos((3*pi/4).*t{ii}); %INSERT ANSWER HERE%  sampled signal
H{ii} = (w>=-2*pi) - (w>=2*pi); %INSERT ANSWER HERE%  ideal filter (in frequency)

ii = ii + 1;
t{ii} = [-2,0,2]; %INSERT ANSWER HERE%  sampled time array
x{ii} = [1,1,1]; %INSERT ANSWER HERE%  sampled signal
H{ii} = (w>=-0.5*pi) - (w>=0.5*pi); %INSERT ANSWER HERE%  ideal filter (in frequency)

ii = ii + 1;
t{ii} = [-3,-2,-1,0,1,2]; %INSERT ANSWER HERE%  sampled time array
x{ii} = [1,1,1,1,1,1]; %INSERT ANSWER HERE%  sampled signal
H{ii} = (w>=-pi) - (w>=pi); %INSERT ANSWER HERE%  ideal filter (in frequency)

ii = ii + 1;
t{ii} = -3:0.25:2.75; %INSERT ANSWER HERE%  sampled time array
x{ii} = ones(24,1); %INSERT ANSWER HERE%  sampled signal
H{ii} = ones(M,1); %INSERT ANSWER HERE%  ideal filter (in frequency)

for ii = 1:length(x)
    X{ii}   = CTFT(x{ii},t{ii},w);
    Xf{ii}  = X{ii}.*H{ii};
    xr1{ii} = ICTFT(X{ii},tnew,w);
    xr2{ii} = ICTFT(Xf{ii},tnew,w);
end

% PLOT
% for ii = 1:length(x)
%     
%     disp(['Question 2 ' char(96+ii)  ' Figures'])
%     figure; 
%     set(gcf,'Units','Inches', 'Position', [0 0 6 6])
%     subplot(3,2,1); stem(t{ii},x{ii}); 
%     xlabel('Time [s]'); ylabel('Amplitude');
%     subplot(3,2,3); plot(w,abs(X{ii})); 
%     xlabel('Angular Frequency [rad/s]'); ylabel('Magnitude');
%     subplot(3,2,4); plot(tnew,real(xr1{ii})); 
%     xlabel('Time [s]'); ylabel('Amplitude');
%     subplot(3,2,5); plot(w,abs(Xf{ii})); 
%     xlabel('Angular Frequency [rad/s]'); ylabel('Magnitude');
%     subplot(3,2,6); plot(tnew,real(xr2{ii})); 
%     xlabel('Time [s]'); ylabel('Amplitude');    
% end




%% 
% 
% *Question 2*: 

%% SETUP - DO NOT EDIT

% DEFINE VARIABLES
T   = 0.1;
f1  = 2000;

%% ANSWER

ii = 1;
fs3{ii} = 0; %INSERT ANSWER HERE%  Sampling frequency
t3{ii} =  0; %INSERT ANSWER HERE%  Time axis
x3{ii} = chirp(t3{ii},f1,T);    %  Chirp Signal

ii = ii + 1;
fs3{ii} = 0; %INSERT ANSWER HERE%  Sampling frequency
t3{ii} =  0; %INSERT ANSWER HERE%  Time axis
x3{ii} = chirp(t3{ii},f1,T);    %  Chirp Signal

ii = ii + 1;
fs3{ii} = 0; %INSERT ANSWER HERE%  Sampling frequency
t3{ii} =  0; %INSERT ANSWER HERE%  Time axis
x3{ii} = chirp(t3{ii},f1,T);    %  Chirp Signal


%% COMPUTE FOURIER TRANSFORMS
for ii = 1:length(x3)
    w3{ii} = 0; %INSERT ANSWER HERE%  Angular frequency
    X3_mag{ii} = 0; %INSERT ANSWER HERE%  Magnitude response
end

% % PLOT
% for ii = 1:length(x3)
%     disp(['Question 3 ' char(96+ii)  ' Figures'])
%     figure; 
%     set(gcf,'Units','Inches', 'Position', [0 0 6 6])
%     subplot(2,1,1); plot(t3{ii},x3{ii}); 
%     xlabel('Time [s]'); ylabel('Amplitude');
%     subplot(2,1,2); plot(w3{ii},X3_mag{ii}); 
%     xlabel('Angular Frequency [rad/s]'); ylabel('Magnitude');
%     ylim([0 max(X3_mag{ii})*1.5+eps])
% end


%% Uncomment the below after completing the above

% disp('Playing Oversampled Chirp')
% soundsc(x3{1}, fs3{1})
% pause(length(x3{1})/fs3{1})
% 
% disp('Playing Critically Sampled Chirp')
% soundsc(x3{2}, fs3{2})
% pause(length(x3{2})/fs3{2})
% 
% disp('Playing Undersampled Sampled Chirp')
% soundsc(x3{3}, fs3{3})
% pause(length(x3{3})/fs3{3})



%% 
% *Supporting Functions (do not change -- only fill in the required code in 
% chirp function):*
%% 
% 

function x = chirp(t,f1,T)

x = 0; %INSERT ANSWER HERE%  
    
end

function X = DTFT(x,n,w)
% DTFT(X,N,W)  computes the Discrete-time Fourier Transform of signal X
% at time indices N across frequencies defined by W. 

    % CHECK IF ROW VECTOR
    ntrx = 0; if size(x,1) == 1, x = x(:); ntrx = 1; end

    X = zeros(length(w),1);
    for nn = 1:length(x)
        X = X + x(nn).*exp(-1j*w(:).*n(nn));
    end
    
    % RETURN VECTOR WITH INPUT ORIENTATION
    if ntrx; X = X.'; end
    
end

function X = CTFT(x,t,w)
% CTFT(X,T,W)  computes the Continuous-time Fourier Transform of signal X
% at time indices N across frequencies defined by W. 

    % CHECK IF ROW VECTOR
    ntrx = 0; if size(x,1) == 1, x = x(:); ntrx = 1; end

    dt = t(2) - t(1);
    X = zeros(length(w),1);
    for nn = 1:length(x)
        X = X + x(nn).*exp(-1j*w(:).*t(nn)).*dt;
    end
    
    % RETURN VECTOR WITH INPUT ORIENTATION
    if ntrx; X = X.'; end

end

function x = ICTFT(X,t,w)
% ICTFT(X,T,W)  computes the inverse continuous-time Fourier Transform
% of signal X at time indices N across frequencies defined by W. 

    % CHECK IF ROW VECTOR
    ntrx = 0; if size(X,1) == 1, X = X(:); ntrx = 1; end

    dw = w(2) - w(1);
    x = zeros(length(t),1);
    for nn = 1:length(X)
        x = x + 1/(2*pi).*X(nn).*exp(1j*w(nn).*t(:)).*dw;
    end
    
    % RETURN VECTOR WITH INPUT ORIENTATION
    if ntrx; x = x.'; end
    
end

function z = synthesize(cnote, dur, A, fs)

    % GET NUMBER OF NOTES
    N = length(cnote);        % Number of characters in name
    
    % CREATE MUSIC
    z = cell2mat(arrayfun(@(n) A(n).*simple_sawtooth(cnote(n), dur(n), fs), 1:N, 'UniformOutput', false )).'; 

end


function z = simple_sawtooth(note, dur, fs)
%SIMPLE_SAWTOOTH  Create a single sawtooth wave note
%   Z = SIMPLE_SAWTOOTH(NOTE,DUR,FS) create a time-modulated sawtooth 
%   note, where NOTE is the note frequency (in Hz), DUR is the note
%   duration (in samples), and FS is the sampling frequency. 
%
%   see also: sawtooth
%

    % BUILD INITIAL SAWTOOTH
    z = sawtooth(2*pi*note*(0:dur-1)/fs,0.2);
    Pattack  = .2;             % Length of attack  (proportion)
    Pdecay   = .2;             % Length of decay   (proportion)
    Prelease = .5;             % Legnth of release (proportion)
    
    Llength  = numel(z);       % Length of signal

    % SET LOW FREQEUNCY SIGNAL TO MODULATE WITH SAWTOOTH
    Lattack  = floor(Llength*Pattack);                      % Length of attack
    Ldecay   = floor(Llength*Pdecay);                       % Length of decay  
    Lrelease = floor(Llength*Prelease);                     % Legnth of release
    Lsustain = ceil(Llength - Lattack - Ldecay - Lrelease); % length of sustain

    Vattack  = 0.8;                                           % Attack maximum value
    Vsustain = 0.5;                                         % Sustain value

    attack  = linspace(0, Vattack, Lattack);                % Attack time weights
    decay   = linspace(Vattack, Vsustain, Ldecay);          % Deacy time weights
    sustain = linspace(Vsustain, Vsustain, Lsustain);       % Sustain time weights
    release = linspace(Vsustain, 0, Lrelease);              % Release time weights

    weight = [attack,decay,sustain,release];                % Concatenate
    weight = conv(weight,exp(-0.0001*(1:1000)), 'same');    % Smooth everything

    % APPLY MODULATION
    z = z.*weight;                                          % Output signal

end

function pzplot(b,a)
% PZPLOT(B,A)  plots the pole-zero plot for the filter described by
% vectors A and B.  The filter is a "Direct Form II Transposed"
% implementation of the standard difference equation:
% 
%    a(1)*y(n) = b(1)*x(n) + b(2)*x(n-1) + ... + b(nb+1)*x(n-nb)
%                          - a(2)*y(n-1) - ... - a(na+1)*y(n-na)
% 

    % MODIFY THE POLYNOMIALS TO FIND THE ROOTS 
    b  = b(1:find(b,1,'last'));
    a  = a(1:find(a,1,'last'));
    b1 = zeros(max(length(a),length(b)),1); % Need to add zeros to get the right roots
    a1 = zeros(max(length(a),length(b)),1); % Need to add zeros to get the right roots
    b1(1:length(b)) = b;    % New a with all values
    a1(1:length(a)) = a;    % New a with all values

    % FIND THE ROOTS OF EACH POLYNOMIAL AND PLOT THE LOCATIONS OF THE ROOTS
    h1 = plot(real(roots(a1)), imag(roots(a1)));
    hold on;
    h2 = plot(real(roots(b1)), imag(roots(b1)));
    hold off;

    % DRAW THE UNIT CIRCLE
    circle(0,0,1)
    
    % MAKE THE POLES AND ZEROS X's AND O's
    set(h1, 'LineStyle', 'none', 'Marker', 'x', 'MarkerFaceColor','none', 'linewidth', 1.5, 'markersize', 8); 
    set(h2, 'LineStyle', 'none', 'Marker', 'o', 'MarkerFaceColor','none', 'linewidth', 1.5, 'markersize', 8); 
    axis equal;
    
    % DRAW VERTICAL AND HORIZONTAL LINES
    xminmax = xlim();
    yminmax = ylim();
    line([xminmax(1) xminmax(2)],[0 0], 'linestyle', ':', 'linewidth', 0.5, 'color', [1 1 1]*.1)
    line([0 0],[yminmax(1) yminmax(2)], 'linestyle', ':', 'linewidth', 0.5, 'color', [1 1 1]*.1)
    
    % ADD LABELS AND TITLE
    xlabel('Real Part')
    ylabel('Imaginary Part')
    title('Pole-Zero Plot')
    
end


function circle(x,y,r)
% CIRCLE(X,Y,R)  draws a circle with horizontal center X, vertical center
% Y, and radius R. 
%
    
    % ANGLES TO DRAW
    ang=0:0.01:2*pi; 
    
    % DEFINE LOCATIONS OF CIRCLE
    xp=r*cos(ang);
    yp=r*sin(ang);
    
    % PLOT CIRCLE
    hold on;
    plot(x+xp,y+yp, ':', 'linewidth', 0.5, 'color', [1 1 1]*.1);
    hold off;
    
end