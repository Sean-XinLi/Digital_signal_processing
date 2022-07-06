%% 
% 
% 
% *Question 1*: 
%% ANSWER

% SETUP
fs       = 44100;           % Sampling rate [in Hz]
Wc       = 2*pi*(5000);     % Continuous frequency cut-off [in rad/s]
Nimp     = 50;              % Impulse response length
Nfft     = 1000;            % Fourier transform length
nimp     = 0:(Nimp-1);      % Impulse response time
Wfft     = (0:(Nfft-1))/Nfft*fs*2*pi;  % Fourier transform frequencies

% ANSWERS

wc = Wc / fs;

% QUESTION 1(a)

N = 50;
n = 0:(N-1);
wn = 0.5.*(1-cos(2*pi*(n)./(N-1)));
hn_1 = wc/pi.*sinc_func(wc.*(n-(N-1)/2));
hn = hn_1 .* wn;


b{1}     = hn;  %INSERT ANSWER HERE%     Numerator filter coefficients
a{1}     = 1;  %INSERT ANSWER HERE%     Denominator filter coefficients
h_imp{1} = hn;  %INSERT ANSWER HERE%     Impulse reponse

H_mag{1} = abs(fft(hn,Nfft));  %INSERT ANSWER HERE%     Magnitude response
H_phs{1} = angle(fft(hn,Nfft));  %INSERT ANSWER HERE%     Phase response


% QUESTION 1(b)
N = 50;
Hn_2 = zeros(size(N));
for i=1:(N+1)
    Hn_2(i) = (-1)^i;   
end

for i=1:(N+1)
    if(i > (25/4))
        Hn_2(i) = 0;
    end
end
hn_2 = ifft(Hn_2,N);

b{2}     = hn_2;  %INSERT ANSWER HERE%     Numerator filter coefficients
a{2}     = 1;  %INSERT ANSWER HERE%     Denominator filter coefficients
h_imp{2} = hn_2;  %INSERT ANSWER HERE%     Impulse reponse
H_mag{2} = abs(fft(hn_2,Nfft));  %INSERT ANSWER HERE%     Magnitude response
H_phs{2} = angle(fft(hn_2,Nfft));  %INSERT ANSWER HERE%     Phase response

% QUESTION 1(c)

omega_z = 20; %INSERT ANSWER HERE% 
omega_p = 0.8; %INSERT ANSWER HERE% 
Q_p = 5; %INSERT ANSWER HERE% 
Q_z = 2; %INSERT ANSWER HERE% 

sz1 = omega_z*(1/Q_z - sqrt(1/(Q_z^2)-1));
sz2 = omega_z*(1/Q_z + sqrt(1/(Q_z^2)-1));
sp1 = omega_p*(1/Q_p - sqrt(1/(Q_p^2)-1));
sp2 = omega_p*(1/Q_p + sqrt(1/(Q_p^2)-1));

z1 = 1/(1+sz1);
z2 = 1/(1+sz2);
p1 = 1/(1+sp1);
p2 = 1/(1+sp2);

p = [p1,p2];
z = [z1,z2];

[b_3,a_3] = pz2ba(p,z);

b{3}     = b_3;  %INSERT ANSWER HERE%     Numerator filter coefficients
a{3}     = a_3;  %INSERT ANSWER HERE%     Denominator filter coefficients
x = (n==0);
hn_3 = filter(b_3,a_3,x);
h_imp{3} = hn_3;  %INSERT ANSWER HERE%     Impulse reponse
H_mag{3} = abs(fft(hn_3,Nfft));  %INSERT ANSWER HERE%     Magnitude response
H_phs{3} = angle(fft(hn_3,Nfft));  %INSERT ANSWER HERE%     Phase response



% QUESTION 1(d)

sqwaves = cell(2,1);
filtered_waves1 = cell(3,1);
filtered_waves2 = cell(3,1);
t = ((1./Nimp).*(0:Nfft-1))';
sqwaves{1} = square(2*pi*0.4*t);
sqwaves{2} = square(2*pi*1*t);


 for jj = 1:3
    filtered_waves1{jj} = wc/pi*sinc_func(wc*(n))*0.5.*(1-cos(2*pi*(n+(N-1)/2)/(N-1))).*sqwaves{1}./hn(1); %Insert Answer Here 
    filtered_waves2{jj} = wc/pi*sinc_func(wc*(n))*0.5.*(1-cos(2*pi*(n+(N-1)/2)/(N-1))).*sqwaves{2}./hn(1); %Insert Answer Here
 end



% PLOT RESULTS
for mm = 1:3
    
    figure;
    pzplot(b{mm},a{mm});

    figure;
    subplot(311)
    stem(nimp, -h_imp{mm})
    subplot(312)
    plot(Wfft, H_mag{mm})
    subplot(313)
    plot(Wfft, H_phs{mm})
    
        
    figure;
    plot(sqwaves{1}(1:ceil(length(t)/5)))
    title('Actual Square Wave - 1')
    
    figure;
    plot(filtered_waves1{mm}(1:ceil(length(t)/5)))
    title('Filtered Signal')
    
    figure;
    plot(sqwaves{2}(1:ceil(length(t)/5)))
    title('Actual Square Wave - 2')
    
    figure;
    plot(filtered_waves2{mm}(1:ceil(length(t)/5)))
    title('Filtered Signal')
    
end


%% 
% ========================= SUPPORTING FUNCTIONS =========================

function x = sinc_func(n)
    x = sin(n)./(n); x(n==0) = 1;
end

function x = chirp(t,f1,T)
    x = cos(2.*pi.*((f1/(2*T)).*t.^2));    
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

function s = square(t)

    s = 2*(mod(t,2*pi) < pi)-1;

end