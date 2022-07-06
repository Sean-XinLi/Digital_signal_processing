clear;

%% QUESTION 1

%% SETUP - DO NOT EDIT
n = 0:20;
N = length(n);
w = -pi:pi/20:pi;
fs = 2000;
t = 1/fs:1/fs:1;


%% ANSWER

% DEFINE YOUR INPUT
xi = zeros(N,1);  xi(1) = 1;   %REPLACE WITH YOUR ANSWER%


% DEFINE YOUR FILTER COEFFICIENTS
%   NOTE: WE CHANGED THE NOTATION SLIGHTLY TO BE EASIER 
%         TO DESIGN THE TEMPLATE AND HAVE LESS TYPOS
 
ii = 1;
b1{ii} = [1,1,1,1,1];  %REPLACE WITH YOUR ANSWER%
a1{ii} = 1;   %REPLACE WITH YOUR ANSWER%

ii = ii + 1;
b1{ii} = [1,-1];  %REPLACE WITH YOUR ANSWER%
a1{ii} = 1;   %REPLACE WITH YOUR ANSWER%

ii = ii + 1;
b1{ii} = 1;  %REPLACE WITH YOUR ANSWER%
a1{ii} = [1,0.75];   %REPLACE WITH YOUR ANSWER%

ii = ii + 1;
[Ha_1,Hb_1] = ba2pz([1,-1],[1,1,1,1,1]);
H_1 = [Ha_1;Hb_1(4)];
[bb,aa] = pz2ba([0.75*exp(1j*(pi/4)),0.75*exp(1j*(-pi/4))],H_1);
b1{ii} = 1;  %REPLACE WITH YOUR ANSWER%
a1{ii} = aa;   %REPLACE WITH YOUR ANSWER%

ii = ii + 1;
b1{ii} = bb;  %REPLACE WITH YOUR ANSWER%
a1{ii} = aa;   %REPLACE WITH YOUR ANSWER%


% COMPUTE IMPULSE RESPONSE AND FOURIER TRANSFORM
%   NOTE: WE DECIDED TO REDUCE CODE CLUTTER BY BUILDING FOR-LOOPS FOR PLOTTING
for ii = 1:length(b1)
    h1{ii} = filter(b1{ii},a1{ii},xi); %REPLACE WITH YOUR ANSWER%
    H1_mag{ii} = abs(DTFT(h1{ii},n,w)); %REPLACE WITH YOUR ANSWER%
    H1_phs{ii} = angle(DTFT(h1{ii},n,w)); %REPLACE WITH YOUR ANSWER%
end

% PLOT
%   NOTE: WE DECIDED TO REDUCE CODE CLUTTER BY BUILDING FOR-LOOPS FOR PLOTTING
for ii = 1:length(b1)
    
    disp(['Question 1 ' char(96+ii)  ' Figures'])
    figure; 
    set(gcf,'Units','Inches', 'Position', [0 0 6 6])
    subplot(2,2,1); stem(n,h1{ii}); 
    xlabel('Time [s]'); ylabel('Amplitude');
    subplot(2,2,2); pzplot(b1{ii}, a1{ii});
    subplot(2,2,3); plot(w,H1_mag{ii}); 
    xlabel('Angular Frequency [rad/s]'); ylabel('Magnitude');
    ylim([0 max(H1_mag{ii})*1.25+0.1])
    subplot(2,2,4); plot(w,H1_phs{ii}); 
    xlabel('Angular Frequency [rad/s]'); ylabel('Phase');
    ylim([-pi pi]);
    
end

%% QUESTION 2

%% SETUP - DO NOT EDIT
n = 0:20;
N = length(n);
w = -pi:pi/20:pi;
fs = 2000;
t = 1/fs:1/fs:1;


%% ANSWER 

% DEFINE YOUR FILTER COEFFICIENTS
ii = 1;
i1 = [1,-1];
i2 = [1, -exp(1j*(pi/4))];
i3 = [1, -exp(-1j*(pi/4))];
i4 = [1,-0.9*exp(-1j)];
i5 = [1,-0.9*exp(1j)];
b2{ii} = conv(conv(i2,i3),i1);  %REPLACE WITH YOUR ANSWER%
a2{ii} = conv(i4,i5); %REPLACE WITH YOUR ANSWER%

ii = ii + 1;
b2{ii} = 1;  %REPLACE WITH YOUR ANSWER%
a2{ii} = [1,-exp(1j*pi/6)-exp(-1j*pi/6)];  %REPLACE WITH YOUR ANSWER%

ii = ii + 1;
b2{ii} = [0,1];  %REPLACE WITH YOUR ANSWER%
a2{ii} = [1,0];  %REPLACE WITH YOUR ANSWER%

% COMPUTE IMPULSE RESPONSE AND FOURIER TRANSFORM
%   NOTE: WE DECIDED TO REDUCE CODE CLUTTER BY BUILDING FOR-LOOPS FOR PLOTTING
for ii = 1:length(b2)
    h2{ii} = filter(b2{ii},a2{ii},xi);
    H2_mag{ii} = abs(DTFT(h2{ii},n,w));
    H2_phs{ii} = angle(DTFT(h2{ii},n,w));
end

% PLOT
%   NOTE: WE DECIDED TO REDUCE CODE CLUTTER BY BUILDING FOR-LOOPS FOR PLOTTING
for ii = 1:length(b2)
    
    disp(['Question 2 ' char(96+ii)  ' Figures'])
    figure; 
    set(gcf,'Units','Inches', 'Position', [0 0 6 6])
    subplot(2,2,1); stem(n,h2{ii}); 
    xlabel('Time [s]'); ylabel('Amplitude');
    subplot(2,2,2); pzplot(b2{ii}, a2{ii});
    subplot(2,2,3); plot(w,H2_mag{ii}); 
    xlabel('Angular Frequency [rad/s]'); ylabel('Magnitude');
    ylim([0 max(H2_mag{ii})*1.25+0.01])
    subplot(2,2,4); plot(w,H2_phs{ii}); 
    xlabel('Angular Frequency [rad/s]'); ylabel('Phase');
    ylim([-pi pi]);
    
end

%% QUESTION 3

%% SETUP - DO NOT EDIT
load('code5_data.mat');    % GIVES US 'message' and 'code' variables



%% ANSWER

% <----
% PUT ANY CODE YOU NEED HERE 

% <----
n = 1:length(message);
N = length(message);
w = -pi:pi/100:pi;

message_new = DTFT(message,n,w);
figure;
plot(w,message_new);
code_re = fliplr(code);
conv1 = conv(message,code_re);
[max,index]= max(conv1);

indx_q3 = index - 100; %REPLACE WITH YOUR ANSWER%

disp(['Index location: Sample ' num2str(indx_q3)])


%% SUPPORTING FUNCTIONS 

function X = DTFT(x,n,w)
% DTFT(X,N,W)  computes the Discrete-time Fourier Transform of signal X
% at time indices N across frequencies defined by W. 

    X = zeros(length(w),1);
    for nn = 1:length(x)
        X = X + x(nn).*exp(-1j*w(:).*n(nn));
    end
    
end

function X = CTFT(x,t,w)
% DTFT(X,T,W)  computes the Continuous-time Fourier Transform of signal X
% at time indices N across frequencies defined by W. 

    dt = t(2) - t(1);
    X = zeros(length(w),1);
    for nn = 1:length(x)
        X = X + x(nn).*exp(-1j*w(:).*t(nn)).*dt;
    end
    
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




function [b,a] = pz2ba(p,z)
% PZ2BA(P,Z)  Converts poles P and zeros Z to filter coefficients
%             B and A
%
% Filter coefficients are defined by:
%    a(1)*y(n) = b(1)*x(n) + b(2)*x(n-1) + ... + b(nb+1)*x(n-nb)
%                          - a(2)*y(n-1) - ... - a(na+1)*y(n-na)
% 

    % CONVERT ROOTS (POLES AND ZEROS) INTO POLYNOMIALS
    b = poly(z);
    a = poly(p);

end


function [p,z] = ba2pz(b,a)
% BA2PZ(B,A)  Converts filter coefficients B and A into poles P and zeros Z
% 
% Filter coefficients are defined by:
%    a(1)*y(n) = b(1)*x(n) + b(2)*x(n-1) + ... + b(nb+1)*x(n-nb)
%                          - a(2)*y(n-1) - ... - a(na+1)*y(n-na)
% 

    % MODIFY THE POLYNOMIALS TO FIND THE ROOTS 
    b1 = zeros(max(length(a),length(b)),1); % Need to add zeros to get the right roots
    a1 = zeros(max(length(a),length(b)),1); % Need to add zeros to get the right roots
    b1(1:length(b)) = b;    % New a with all values
    a1(1:length(a)) = a;    % New a with all values

    % FIND THE ROOTS OF EACH POLYNOMIAL
    p = real(roots(a1))+1j*imag(roots(a1));
    z = real(roots(b1))+1j*imag(roots(b1));
    
end

