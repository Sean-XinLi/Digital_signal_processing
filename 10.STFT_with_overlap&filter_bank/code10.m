%% Question 1
%1a  FUNCTIONS ARE AT THE END OF THE CODE
clear;

% -----------------
% QUESTION 1 SETUP
% -----------------
W = 100;   % Window size for STFT

% AUDIO FILE
[x1, fs] = audioread('urquan.wav');
t0 = 0:W/fs:(length(x1)-1)/fs;
t1 = 0:W/2/fs:(length(x1)-1)/fs;
w  = (0:(W-1))*2*pi/W; 

% -------------------
% QUESTION 1 RESULTS
% -------------------

% QUESTION 1(b)
ans_1b  = stft_woverlap(x1, W);
ans_1b2 = istft_woverlap(ans_1b, W);

figure;
imagesc(t0, w,  10*log10(abs(ans_1b)))
xlabel('Time [s]'); 
ylabel('Norm. Ang. Frequency [rad /s]'); 
title('1(b): STFT Spectrogram in dB (No overlap)')
drawnow;


% QUESTION 1(c)
ans_1c = 'there is no choppiness noise with overlapping, because overlapping make STFT frequency continuous.';

test0_stft = stft_func(x1, W);
test0_stft(abs(test0_stft) < max(abs(test0_stft))*0.5) = 0; 
test0_out  = istft_func(test0_stft, W);
test1_stft = stft_woverlap(x1, W);
test1_stft(abs(test1_stft) < max(abs(test1_stft))*0.5) = 0;
test1_out  = istft_woverlap(test1_stft, W);

figure;
subplot(211)
imagesc(t0, w,  10*log10(abs(test0_stft)))
xlabel('Time [s]'); 
ylabel('Norm. Ang. Frequency [rad /s]'); 
title('1(c): STFT Spectrogram in dB (No overlap)')
subplot(212)
imagesc(t1, w,  10*log10(abs(test1_stft)))
xlabel('Time [s]'); 
ylabel('Norm. Ang. Frequency [rad /s]'); 
title('1(c): STFT Spectrogram in dB (With Overlap)')
drawnow;

% % PLAY AUDIO
disp('Playing original audio...')
soundsc(x1, fs)
pause(length(x1)/fs)

disp('Playing STFT (no overlap) reconstructed audio...')
soundsc(test0_out, fs)
pause(length(test0_out)/fs)

disp('Playing STFT (with overlap) reconstructed audio...')
soundsc(test1_out, fs)
pause(length(test1_out)/fs)



%% Question 2

% -----------------
% QUESTION 1 SETUP
% -----------------
N = 100;            % Number of filters

% AUDIO FILE 
[x, Fs] = audioread('urquan.wav');


% -------------------
% QUESTION 1 RESULTS
% -------------------

% QUESTION #2(a)

% MODIFIED DISCRETE COSINE BASIS
g = zeros(200, N); %INSERT ANSWER HERE
for i = 0:(200-1)
    for j = 0:(N - 1)
    g(i+1,j+1) = 1 / N ^ (0.5) * cos(pi / N *(i + (N + 1 ) / 2) *(j + 1 / 2));
        
    end
end
h = zeros(200, N); %INSERT ANSWER HERE
for i =0:(200-1)
    for j = 0:(N - 1)
        h(i+1,j+1) = g(2*N-i, j+1);
    end
end

% APPLY FILTER BANK
ans_2a_v = fb_analysis(x,h);       % write fb_analysis function

% PLOTS FOR QUESTION 2(a)
t = (0:N:((size(ans_2a_v,1)-1)*N))/Fs;
w = (0:(N-1))*pi/N; 

figure
imagesc(t, w,  10*log10(abs(ans_2a_v.')))
xlabel('Time [s]'); 
ylabel('Norm. Ang. Frequency [rad / s]'); 
title('Q2(a): Filterbank output in dB')
drawnow;


% QUESTION #2(b)
ans_2b_y = fb_synthesis(ans_2a_v,g);      % write your Synthesis Bank function

% PLOTS FOR QUESTION 2(b)
tx = (0:(size(x,1)-1))/Fs;
ty = (0:(size(ans_2b_y,1)-1))/Fs;

figure
subplot(211)
plot(tx, x, 'linewidth', 1);
xlabel('Time [s]')
ylabel('Amplitude')
title('Q2(b): Input Signal')
subplot(212)
plot(ty,ans_2b_y, 'linewidth', 1);
xlabel('Time [s]')
ylabel('Amplitude')
title('Q2(b): Reconstructed Signal')
drawnow;

figure
plot( x(:), 'linewidth', 1);
hold on;
plot(ans_2b_y, 'linewidth', 1);
hold off;
xlabel('Time [s]')
ylabel('Amplitude')
title('Q2(b): Close-up Comparison')
drawnow; 

% QUESTION #2(c)
[max_x,idx_x] = max(x);
[max_y,idx_y] = max(ans_2b_y);
delay = idx_y - idx_x;
ans_2c = delay; %INSERT ANSWER HERE;      

% QUESTION #2(d) 
ans_2d = 'the filters have group delay'; %INSERT ANSWER HERE

% QUESTION #2(e) 
ans_2e = 'differences';



%% 
% ========================= SUPPORTING FUNCTIONS =========================

function v = fb_analysis(x,h)

    % INITIALIZE LENGTHS
    Q = length(x);      % Number of Samples
    M = size(h,2);      % Number of Filters
    N = size(h,1);      % Length of Filters
    
    % WRITE YOUR CODE HERE
    % ==================================================
    y_len = ceil((Q + N - 1)/ M);
    y = zeros(y_len, M);
    for i = 1:M
        temp = conv(x, h(:,i));
        y(:, i) = downsample_func(temp, M);
    end
    v = y;

end

function y = fb_synthesis(v,g)

    % INITIALIZE LENGTHS
    M = size(g,2);           % Number of Filters
    N = size(g,1);           % Length of Filters
    Qp = size(v,1)*M+N-1;    % Number of Samples
    
    % WRITE YOUR CODE HERE
    % ==================================================
    y_sum =zeros(Qp, 1);
    for i = 1:M
        v_up = upsample_func(v(:,i), M);
        y_i = conv(v_up, g(:,i));
        y_sum = y_sum +y_i;
    end
    y = y_sum;

end

function [xSTFT,M] = stft_woverlap(x, W)

    % WRITE YOUR CODE HERE
    N = length(x);          % Length of signal
    M = floor(N/W);         % Number of segments
    STFT = zeros(W, M);     % Initialize short-time Fourier transform
    for m = 2:(2 * M - 1)
        % FIND FREQUENCIES
        z = x((W / 2*(m-1)+1):(W / 2 * m + W / 2));    % Get data segment
        STFT(:,m) = fft(z);                    % Compute Fourier Transform
    end
    xSTFT = STFT;
    
end

function [x] = istft_woverlap(xSTFT, W)

    % WRITE YOUR CODE HERE
    % CHOOSE WINDOW
    M = (size(xSTFT,2) + 1)/2;      % Number of segments
    W = size(xSTFT,1);              % Number of samples
    N = M * W;                      % The length of signal
     % LOOP OVER SEGMENTS
    x_new = zeros(N,1);
    for m = 2:(2 * M - 1) 
        x_new((W/2*(m-1)+1):(W/2*m + W/2)) = x_new((W/2*(m-1)+1):(W/2*m + W/2)) ...
                                             + real(ifft(xSTFT(:,m)) .* hann(W));
    end
    x = x_new;
    
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

end

function y = upsample_func(x,M)

    Q = length(x);
    y = zeros(M*Q,1);
    y(1:M:(M*Q)) = x;

end

function y = downsample_func(x,N)

    Q = length(x);
    y = x(1:N:Q);

end

function y = downsample_antialias_func(x,N)

    Q = length(x);
    x = lpf_func(x,pi/N,10);
    y = x(1:N:Q);

end

function y = upsample_interp_func(x,M)

    Q = length(x);
    y = zeros(M*Q,1);
    y(1:M:(M*Q)) = x;
    y = M*lpf_func(y,pi/M,10);

end

function x = sinc_func(n)
    x = sin(n)./(n); x(n==0) = 1;
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
