%% *Question 1*

% FUNCTIONS ARE STARTED AT END OF FILE

%% Question 2

% -----------------
% QUESTION 2 SETUP
% -----------------

% CHIRP SIGNAL
Nx   = 256;
Nfft = 1024;
nx   = 0:(Nx-1);
w    = 2*pi*(0:(Nfft-1))/Nfft;
f1   = 1/8;
x2   = chirp(nx, f1, Nx); 

figure();
subplot(211)
stem(nx, x2,'filled', 'markersize', 2)
hold on; plot(nx, x2); hold off;
xlim([0 max(nx)+eps])
xlabel('Samples')
ylabel('Amplitude')
title('INITIAL CHIRP FOR DOWNSAMPLING');

subplot(212)
plot(w, abs(fft(x2,Nfft)))
xlim([0 2*pi])

% -----------------
% QUESTION 2 ANSWERS
% -----------------

% QUESTION 2(a)%
x1_d = downsample_func(x2, 2);
n1 = 0: (length(x1_d) - 1);
ans_2{1} = x1_d;    %INSERT ANSWER HERE%   
ans_n2{1} = n1;    %INSERT ANSWER HERE% (n indices for result)

% QUESTION 2(b)
x2_d = downsample_func(x2, 5);
n2 = 0: (length(x2_d) - 1);
ans_2{2} = x2_d;    %INSERT ANSWER HERE%   
ans_n2{2} = n2;    %INSERT ANSWER HERE% (n indices for result)

% QUESTION 2(c)

x3_d = downsample_antialias_func(x2, 5);
n3 = 0: (length(x3_d) - 1);
ans_2{3} = x3_d;    %INSERT ANSWER HERE%   
ans_n2{3} = n3;    %INSERT ANSWER HERE% (n indices for result)

% PLOT RESULTS
for ii = 1:3
    
    figure(); 
    subplot(211)
    stem(ans_n2{ii}, ans_2{ii},'filled', 'markersize', 2)
    hold on; plot(ans_n2{ii}, ans_2{ii}); hold off;
    xlabel('Samples')
    xlim([0 max(ans_n2{ii})+eps])
    title(['QUESTION 1(' char(96+ii) ') FIGURES'])
    
    subplot(212)
    plot(w, abs(fft(ans_2{ii},Nfft)))
    xlabel('Normalized Frequency [rad/s]')
    xlim([0 2*pi])
    ylim([0 max(30,max(abs(fft(ans_2{ii},Nfft))))+eps])
    
end


%% Question 3

% -----------------
% QUESTION 3 SETUP
% -----------------

% CHIRP SIGNAL
Nx   = 64;
Nfft = 1024;
nx   = 0:(Nx-1);
w    = 2*pi*(0:(Nfft-1))/Nfft;
f1   = 1/8;
x3    = chirp(nx, f1, Nx); 

figure();
subplot(211)
stem(nx, x3,'filled', 'markersize', 2)
hold on; plot(nx, x3); hold off;
xlim([0 max(nx)+eps])
xlabel('Samples')
ylabel('Amplitude')
title('INITIAL CHIRP FOR UPSAMPLING');

subplot(212)
plot(w, abs(fft(x3,Nfft)))
xlim([0 2*pi])

% -----------------
% QUESTION 3 ANSWERS
% -----------------

% QUESTION 3(a)
x1_up = upsample_func(x3, 2);
n4 = 0: (length(x1_up) - 1);
ans_3{1} = x1_up;    %INSERT ANSWER HERE%   
ans_n3{1} = n4;    %INSERT ANSWER HERE% (n indices for result)

% QUESTION 3(b)
x2_up = upsample_func(x3, 5);
n5 = 0: (length(x2_up) - 1);
ans_3{2} = x2_up;    %INSERT ANSWER HERE%   
ans_n3{2} = n5;    %INSERT ANSWER HERE% (n indices for result)

% QUESTION 3(c)
a= length(x3);
x3_up = upsample_interp_func(x3, 5);
n6 = 0: (length(x3_up) - 1);
ans_3{3} = x3_up;    %INSERT ANSWER HERE%   
ans_n3{3} = n6;    %INSERT ANSWER HERE% (n indices for result)

% PLOT RESULTS
for ii = 1:3
    
    figure(); 
    subplot(211)
    stem(ans_n3{ii}, ans_3{ii},'filled', 'markersize', 2)
    hold on; plot(ans_n3{ii}, ans_3{ii}); hold off;
    xlabel('Samples')
    xlim([0 max(ans_n3{ii})+eps])
    title(['QUESTION 1(' char(96+ii) ') FIGURES'])
    
    subplot(212)
    plot(w, abs(fft(ans_3{ii},Nfft)))
    xlabel('Normalized Frequency [rad/s]')
    xlim([0 2*pi])
    ylim([0 max(30,max(abs(fft(ans_3{ii},Nfft))))+eps])
    
    
end
 


%% *Question 4*

% -----------------
% QUESTION 4 SETUP
% -----------------

% SETUP
W = 600;   % Window size for STFT

% AUDIO FILE
[x4, fs] = audioread('urquan.wav');
t = 0:1/fs:(length(x4)-1)/fs;
f = (0:(length(x4)-1))/length(x4)*fs;
N = 3;      % Downsample rate


% -----------------
% QUESTION 4 RESULTS
% -----------------

% QUESTION 4(a)
x_d = downsample(x4, 3);
ans_4a = x_d;   %INSERT ANSWER HERE%     downsampled audio   

% QUESTION 4(b)
stft = stft_func(x4, W);
sample = downsample_func(stft(1,:),3);
stft_d = zeros(W, length(sample));
for i = 1:W
    stft_d(i,:) = downsample_func(stft(i, :), 3);
end
x_re = istft_func(stft_d, W);
ans_4b  = stft_d;  %INSERT ANSWER HERE%     stft of downsampled audio
ans_4b2 = x_re;  %INSERT ANSWER HERE%     istft of downsampled audio

% QUESTION 4(c)
ans_4c = 'for the audio with downsampling ,shrink time but the frequency never overlaped(no aliasing),for downsampling the STFT, shrink time but each frequencies overlaped(aliasing)';


% PLAY AUDIO
disp('Playing original audio...')
soundsc(x4, fs)
pause(length(x4)/fs)

disp('Playing downsampled audio...')
soundsc(ans_4a, fs)
pause(length(ans_4a)/fs)

disp('Playing STFT downsampled audio...')
soundsc(ans_4b2, fs)
pause(length(ans_4b2)/fs)

%% 
% *========================= SUPPORTING FUNCTIONS =========================*

function y = upsample_func(x,M)
    n = length(x);
    x_up = zeros(M * n,1);
    for i = 1:n
        x_up(1 + (i - 1) * M) = x(i);
    end
    
    y = x_up; % REPLACE THIS

end

function y = downsample_func(x,N)
    n = ceil(length(x)/ N);
    x_d = zeros(n,1);
    for i = 1:n
        x_d(i) = x(1+ (i-1) * N);
    end
    y = x_d; % REPLACE THIS

end

function y = downsample_antialias_func(x,N)
    p = 10;
    wc = pi / N;
    x_new = lpf_func(x, wc, p);
    x_d = downsample_func(x_new, N);

    y = x_d; % REPLACE THIS

end

function y = upsample_interp_func(x,M)
    x_up = upsample_func(x,M);
    p = 10;
    wc = pi / M;
    x_up_new = lpf_func(x_up, wc, p);

    y = x_up_new; % REPLACE THIS

end

function x = chirp(t,f1,T)
    x = cos(2.*pi.*((f1/(2*T)).*t.^2));    
end

function y = lpf_func(x,wc,P)

    % COMPUTE Z-TRANSFORM POLE LOCATIONS
    zp = zeros(P,1);
    for k = 1:P
        zp(k) = (1 - wc*exp(-1j*(pi*(2*k-1)/(2*P)+ pi/2)))./abs((1 - wc*exp(-1j*(pi*(2*k-1)/(2*P)+ pi/2))))^2;
    end
    
    % CONVERT POLES AND ZEROS TO B AND A
    [bz,az] = pz2ba(zp,zeros(length(zp),1));
    
    % NORMALIZE 
    bz = bz.*prod(1-zp);
    
    % APPLY FILTER
    y  = real(filter(bz,az,x)); % Real is used to remove round-off errors

end


function [STFT, M] = stft_func(x,W)

    % GET TIME PARAMETERS
    N = length(x);                          % Length of signal
    M = floor(N/W);                      % Number of segments

    % LOOP OVER SEGMENTS
    STFT = zeros(W, M);                     % Initialize short-time Fourier transform
    for m = 1:M
        % FIND FREQUENCIES
        z = x((W*(m-1)+1):(W*m));           % Get data segment
        STFT(:,m) = fft(z);                 % Compute Fourier Transform
    end
end

function x = istft_func(STFT, W)

    % CHOOSE WINDOW
    W       = size(STFT,1);         % Number of samples
    M       = size(STFT,2);         % Number of segments

    % LOOP OVER SEGMENTS
    x = zeros(M*W,1);
    for m = 1:M
        x((W*(m-1)+1):(W*m)) = ifft(STFT(:,m));         % Compute Fourier Transform 
    end
    x = real(x);    % Force real to deal with rounding errors

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
% CTFT(X,T,W)  computes the continuous-time Fourier Transform of signal X
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