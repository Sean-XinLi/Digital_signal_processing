
%% 
% *Question 1*: 

% NO SETUP


%% ANSWER

% QUESTION 1(a)
Nx = 1000000; %INSERT ANSWER HERE% 
Nh = 100000; %INSERT ANSWER HERE% 
f1 = 1/8000; %INSERT ANSWER HERE% 
nx = 0: (Nx - 1); %INSERT ANSWER HERE% 
nh = 0: (Nh - 1); %INSERT ANSWER HERE%

x1a = cos(2 * pi * (f1 / (2 * Nx)) * nx .^ 2); %INSERT ANSWER HERE FOR x[n]% 
h1a = cos(2 * pi *(f1 / (2 * Nh) * (nh + 5000) .^2)); %INSERT ANSWER HERE FOR h[n]% 

figure(1)
subplot(211)
plot(nx,x1a)
xlabel('Samples');
ylabel('Amplitude');
subplot(212)
plot(nh,h1a)
xlabel('Samples');
ylabel('Amplitude');


% QUESTION 1(b)
nyb = 0:(Nx - 1 + Nh - 1); %INSERT ANSWER HERE% 

tic;
y1b = conv(x1a, h1a); %INSERT ANSWER HERE% 
tm1 = toc;

figure(2)
plot(nyb,y1b)
xlabel('Samples');
ylabel('Amplitude');


% QUESTION 1(c)
nyc = 0:(Nx - 1); %INSERT ANSWER HERE% 
h1c = [h1a zeros(1, (Nx - Nh))]; %INSERT ANSWER HERE% 

tic;
y1c = ifft(fft(x1a) .* fft(h1c)) ; %INSERT ANSWER HERE% 
tm2 = toc;

figure(3)
plot(nyc,y1c)
xlabel('Samples');
ylabel('Amplitude');


% % Ungraded questions
% % QUESTION 1(d)
ans_time_conv = tm1; %INSERT ANSWER HERE% 
ans_time_fft = tm2; %INSERT ANSWER HERE% 
% 
% 
% % QUESTION 1(e)
% ans_explain = 'Answer to 1(e) here'; %INSERT ANSWER HERE% 


%% 
% *Question 2*: 

% SETUP -- DO NOT CHANGE
Xstft = cell(4,1);
nx    = cell(4,1);
fx    = cell(4,1);
Nx    = 2560;
f1    = 1/2;


% QUESTION 2(b)
nx1 = 0: (Nx - 1);
x  = chirp(nx1, f1, Nx);  %INSERT ANSWER HERE x[n] FOR THIS QUESTION% 

figure(1);
plot(0:(Nx-1),x)
xlabel('Samples')
ylabel('Amplitude')


% QUESTION 2(c) 
W = [10 40 160 640];

Xstft{1} = stft_func(x,W(1)); 

nx{1} = 0: W(1): (Nx - W(1));  %INSERT ANSWER HERE%     Time-domain Axis
fx{1} = 0: (2 * pi / W(1)) : (2 * pi - 2 * pi / W(1));  %INSERT ANSWER HERE%     Frequency-domain Axis

Xstft{2} = stft_func(x,W(2));  

nx{2} = 0: W(2): (Nx - W(2));  %INSERT ANSWER HERE%     Time-domain Axis
fx{2} = 0: (2 * pi / W(2)): (2 * pi - 2 * pi / W(2));  %INSERT ANSWER HERE%     Frequency-domain Axis

Xstft{3} = stft_func(x,W(3)); 

nx{3} = 0: W(3): (Nx - W(3));  %INSERT ANSWER HERE%     Time-domain Axis
fx{3} = 0: (2 * pi / W(3)):(2 * pi - 2 * pi / W(3));  %INSERT ANSWER HERE%     Frequency-domain Axis

Xstft{4} = stft_func(x,W(4)); 

nx{4} = 0: W(4): (Nx - W(4));  %INSERT ANSWER HERE%     Time-domain Axis
fx{4} = 0: (2 * pi / W(4)):(2 * pi - 2 * pi / W(4));  %INSERT ANSWER HERE%     Frequency-domain Axis

for ii = 1:4
    figure(2);
    set(gcf, 'Units', 'normalized', 'Position', [.1 .1 .4 .8]);
    subplot(2,2,ii)
    imagesc(nx{ii}, fx{ii}, abs(Xstft{ii}))
    ylabel('Normalized Angular Frequency')
    xlabel('Time [samples]')
    axis xy;
end


%% 
% =========================
% SUPPORTING FUNCTIONS
% =========================
%%
function x = chirp(n,f1,N)
    x = cos(2.*pi.*((f1/(2*N)).*n.^2));    
end


function Xstft = stft_func(x,W)
% STFT(X,W)  computes the Short-Time Fourier Transform of signal X
% for window length W 

    % GET TIME PARAMETERS
    N = length(x);                          % Length of signal
    M = floor(N/W);                      % Number of segments

    %% REPLACE THE NEXT LINK WITH THE REST OF THE FUNCTION (TO BE COMPLETED)
    for i = 1 : M
        y = fft(x(((i - 1) .* W + 1) : (i .* W)));
        Xstft(:, i) = y;
    end
    
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