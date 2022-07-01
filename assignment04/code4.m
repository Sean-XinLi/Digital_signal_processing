
%% CLARIFICATIONS ON COMMON ISSUES
% If an answer is assigned to a variable, make sure not to create a 
%   variable with the same name later in your code. The auto-grader will 
%   only see the most recent value.
% When submitting, make sure to upload every file required to make your 
%   code run, often including WAV and MAT files in addition to your M or 
%   MLX files. You can upload all files individually or upload a ZIP with 
%   everything. Uploading extra files will \emph{not} hurt you.


%% ============== Question 1 ==============
% Finish the CTFT and DTFT functions declared later in this file, located
% beneath Question 3. The autograder will grade these functions using a
% SECRET set of predetermined inputs and expected outputs ("test cases").

%% ============== Question 2 ==============
% The following variables are provided for you...
n = -10:10;
t = -10:0.01:10;
w = -4*pi:pi/20:4*pi;

% Construct each of the input signals in the variables below.
% part a
x_t_a = (t>=-3)&(t<3);
x_n_a = (n>=-3)&(n<4);

% part b
x_t_b = (t>=0)&(t<6);
x_n_b = (n>=0)&(n<7);



% part c
x_t_c = ((t>=-3)&(t<3)) .* cos(pi.*t/4);
x_n_c = ((n>=-3)&(n<4)) .* cos(pi.*n/4);

% Calculating outputs using your functions and your constructed inputs...
% No need to edit these.
% part a
X_t_a = CTFT(x_t_a, t, w);
X_n_a = DTFT(x_n_a, n, w);
% part b
X_t_b = CTFT(x_t_b, t, w);
X_n_b = DTFT(x_n_b, n, w);
% part c
X_t_c = CTFT(x_t_c, t, w);
X_n_c = DTFT(x_n_c, n, w);

% part d: Leave a comment describing the key difference that you see
% between CTFT and DTFT results.

% COMMENT HERE :)
% fisrt, according to the amplitude figures
% CTFT signal is continuous time
% DTFT signal is dicrete time
% second, according to the real amplitude figures
% CTFT signal is aperiodic
% DTFT signal is periodic
% third, according to the imaginary amplitude figures
% CTFT signal is aperiodic
% there is no DTFT signal in imaginary amplitude figure
% fourth, according to the magnitude figures
% CTFT signal is aperiodic
% DTFT signal is periodic
% fifth, according to the phase
% CTFT signal is centrosymmetric
% DTFT signal is y-axis symmetry

% Plotting your inputs & outputs...
% No need to edit this.

%disp('Question 1(a) Figures (Continuous-Time)')
figure(1); 
set(gcf,'Units','Inches', 'Position', [0 0 6 6], 'Name', 'Q1(a) Figures (Continuous-Time)', 'NumberTitle', 'off')
subplot(3,2,1); plot(t,(x_t_a)); 
xlabel('Time [s]'); ylabel('Amplitude');
subplot(3,2,3); plot(w,real(X_t_a)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Real Ampltidue');
subplot(3,2,4); plot(w,imag(X_t_a)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Imaginary Ampltidue');
subplot(3,2,5); plot(w,abs(X_t_a)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Magnitude');
subplot(3,2,6); plot(w,angle(X_t_a)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Phase');
ylim([-pi pi]);

%disp('Question 1(a) Figures (Discrete-Time)')
figure(2); 
set(gcf,'Units','Inches', 'Position', [0 0 6 6],'Name','Q1(a) Figures (Discrete-Time)','NumberTitle','off')
subplot(3,2,1); stem(n,(x_n_a)); 
xlabel('Time [samples]'); ylabel('Amplitude');
subplot(3,2,3); plot(w,real(X_n_a)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Real Ampltidue');
subplot(3,2,4); plot(w,imag(X_n_a)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Imaginary Ampltidue');
subplot(3,2,5); plot(w,abs(X_n_a)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Magnitude');
subplot(3,2,6); plot(w,angle(X_n_a)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Phase');
ylim([-pi pi]);

%disp('Question 1(b) Figures (Continuous-Time)')
figure(3); 
set(gcf,'Units','Inches', 'Position', [0 0 6 6], 'Name', 'Q1(b) Figures (Continuous-Time)', 'NumberTitle','off')
subplot(3,2,1); plot(t,(x_t_b)); 
xlabel('Time [s]'); ylabel('Amplitude');
subplot(3,2,3); plot(w,real(X_t_b)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Real Ampltidue');
subplot(3,2,4); plot(w,imag(X_t_b)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Imaginary Ampltidue');
subplot(3,2,5); plot(w,abs(X_t_b)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Magnitude');
subplot(3,2,6); plot(w,angle(X_t_b)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Phase');
ylim([-pi pi]);

%disp('Question 1(b) Figures (Discrete-Time)')
figure(4); 
set(gcf,'Units','Inches', 'Position', [0 0 6 6], 'Name', 'Q1(b) Figures (Discrete-Time)', 'NumberTitle','off')
subplot(3,2,1); stem(n,(x_n_b)); 
xlabel('Time [samples]'); ylabel('Amplitude');
subplot(3,2,3); plot(w,real(X_n_b)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Real Ampltidue');
subplot(3,2,4); plot(w,imag(X_n_b)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Imaginary Ampltidue');
subplot(3,2,5); plot(w,abs(X_n_b)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Magnitude');
subplot(3,2,6); plot(w,angle(X_n_b)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Phase');
ylim([-pi pi]);

%disp('Question 1(c) Figures (Continuous-Time)')
figure(5); 
set(gcf,'Units','Inches', 'Position', [0 0 6 6], 'Name', 'Q1(c) Figures (Continuous-Time)', 'NumberTitle','off')
subplot(3,2,1); plot(t,(x_t_c)); 
xlabel('Time [s]'); ylabel('Amplitude');
subplot(3,2,3); plot(w,real(X_t_c)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Real Ampltidue');
subplot(3,2,4); plot(w,imag(X_t_c)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Imaginary Ampltidue');
subplot(3,2,5); plot(w,abs(X_t_c)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Magnitude');
subplot(3,2,6); plot(w,angle(X_t_c)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Phase');
ylim([-pi pi]);

%disp('Question 1(c) Figures (Discrete-Time)')
figure(6);
set(gcf,'Units','Inches', 'Position', [0 0 6 6], 'Name', 'Q1(c) Figures (Discrete-Time)', 'NumberTitle','off')
subplot(3,2,1); stem(n,(x_n_c)); 
xlabel('Time [samples]'); ylabel('Amplitude');
subplot(3,2,3); plot(w,real(X_n_c)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Real Ampltidue');
subplot(3,2,4); plot(w,imag(X_n_c)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Imaginary Ampltidue');
subplot(3,2,5); plot(w,abs(X_n_c)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Magnitude');
subplot(3,2,6); plot(w,angle(X_n_c)); 
xlabel('Angular Frequency [rad/s]'); ylabel('Phase');
ylim([-pi pi]);

%% ============== Question 3 Part A ==============
% Loading data from the cosine wav file: No need to edit this.
% Don't worry about understanding this part right now...
% But for the curious, fs is the music file's "sampling frequency." :)
[cos_x, fs] = audioread('cosine.wav');
t = 0:1/fs:(length(cos_x)-1)/fs;
w = -2*pi*1000:2*pi:2*pi*1000;

% Use your CTFT function to identify the cosine's positive cyclic
% frequency.

% YOUR CODE HERE.
% Hint: If you don't know where to start, try plotting your CTFT output.
X_i = CTFT(cos_x', t, w);
X = abs(X_i);
x = find(X ==max(X(:)));
w_1 = (x(2) - x(1)) / 2 * pi * 2;

figure(7);
plot(w,X);
xlabel('w');
ylabel('X');
title('question3(a)');

cos_f = w_1/(2*pi);
% cos_f = 3769.91/(2*pi); %answer to part A



%% ============== Question 3 Part B ==============
% Loading data from the music wav file: No need to edit this.
[music_x, fs] = audioread('rolemusic.wav');
t = 0:1/fs:(round(length(music_x)/50)-1)/fs;
w = -2*pi*1000:2*pi:2*pi*1000;

% Calculate the cyclic frequency and amplitude of each note in the music.
% The music consists of 50 evenly spaced notes of equal length (this
% knowledge can be used to isolate each note).

% YOUR CODE HERE.

% segment the signal into 50 evenly space

music_f = zeros(1,50); %cyclic frequency of each note in order
music_A = zeros(1,50); %amplitude of each note in order

period = floor(length(music_x)/50);
for i = 1:50
    music_x_i = music_x(1 + (i - 1) * period : period * i);
    
    X_i = CTFT(music_x_i', t, w);
    X = abs(X_i);
    A = max(X);
    x = find(X ==max(X(:)));
    w_1 = (x(2)-x(1))/2 * pi * 2;
    f = w_1/(2* pi);
    music_f(i) = f;
    music_A(i) = A;
end


% music_f = zeros(1,50); %cyclic frequency of each note in order
% music_A = zeros(1,50); %amplitude of each note in order

% You can listen to your answers by uncommenting the 2 lines of code below.
% WARNING: the soundsc function can be very loud.

%z = synthesize(music_f, round(length(music_x)/50).*ones(length(music_f),1), music_A, fs);
%soundsc(z,fs);


%% ============== Functions for Question 1 ==============
% Don't edit this next line: it enables the autograder to see your functions.
CTFT_handle = @CTFT; DTFT_handle = @DTFT;
% Edit these functions below!

function X=DTFT(x,n,w)
    % function to approximate the DTFT.
    % INPUTS:
    % - x: the signal we are taking the DTFT of. Vector of any length.
    % - n: the value of n at which each sample in x is located. Same length
    %      as x.
    % - w: the omega values at which the output is desired. vector of any
    %      length.
    % OUTPUTS:
    % - X: the DTFT of x at each value of w. Same length as w.
    
    %YOUR CODE BELOW
    % X = zeros(1,length(w));
    X = x * exp((-1j) .* (n' .* w)); %placeholder output; change this
end

function X=CTFT(x,t,w)
    % function to approximate the CTFT.
    % INPUTS:
    % - x: the signal we are taking the DTFT of. Vector of any length.
    % - t: the value of t at which each sample in x is located. Same length
    %      as x.
    % - w: the omega values at which the output is desired. vector of any
    %      length.
    % OUTPUTS:
    % - X: the CTFT of x at each value of w. Same length as w.
    
    %YOUR CODE BELOW
    % X = zeros(1,length(w)); %placeholder output; change this
    X = x * exp((-1j) .* (t' .* w));
end




%% ============== Provided Support Functions ==============
%(Please don't edit these)

function z = synthesize(cnote, dur, A, fs)

    % GET NUMBER OF NOTES
    N = length(cnote);
    
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
