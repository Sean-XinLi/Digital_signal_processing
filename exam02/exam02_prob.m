
clear;

% COMMENT: 
% MAKE SURE THE FOLLOWING FILES ARE IN YOUR WORKING DIRECTORY:
%   ditty1.wav through ditty26.wav
% THIS IS USED FOR MULTIPLE QUESTIONS

% FUNCTIONS PROVIDED AT THE END FROM PRIOR ASSIGNMENTS: 
% upsample_func
% upsample_interp_func
% downsample_func
% downsample_antialias_func
% sinc_func
% chirp
% lpf_func
% stft_woverlap
% istft_woverlap
% DTFT
% CTFT
% ICTFT
% pzplot
% pz2ba
% ba2pz
% wb_analysis
% wb_synthesis
% fb_analysis
% fb_synthesis

%%
% QUESTION 1
[Ta, wb, Hb, Lb, hc, wc] = exam02_q1('12079483');

% QUESTION 1(a)
Ta = abs(Ta);
ans_1a_b = [1/Ta, -1/Ta];   % Derivative Approximation (numerator 'b' filter coefficients)
ans_1a_a = [1];   % Derivative Approximation (denominator 'a' filter coefficients)

% QUESTION 1(b)
o_1b = find(wb==0);
Hb_new_part1 = Hb(1:o_1b-1);
Hb_new_part2= Hb(o_1b:end);
Hb_new = [Hb_new_part2,Hb_new_part1];
wb_new = wb + pi;
H_temp = zeros(1,Lb);
index = ones(1,Lb);
h_1b = zeros(1,Lb);
for n = 1: (Lb-1)
    index(n+1) = 1 + round(length(Hb_new) / Lb * n);
end
X_1b = Hb_new.*exp(1j*wb_new);
for i = 1:Lb
    H_temp(i) = X_1b(index(i));
end
for ii = 1:Lb
    sum_1b = 0;
    for iii = 1:((Lb-1)/2)
        sum_1b = sum_1b + (-1)^(iii)*H_temp(iii+1)*cos(2*pi/Lb*((ii-1)+1/2)*iii);
    end
    h_1b(ii) = 1/Lb*(H_temp(1) + 2 * sum_1b);
end

w = -pi:0.01*pi:(pi-0.01*pi);
n = -7:7;
c = DTFT(h_1b,n,w);
figure(1);
plot(w,abs(c));
figure(2);
stem(n,h_1b);
figure(10);
plot(wb,Hb);

ans_1b = h_1b;   % An FIR Filter

% QUESTION 1(c)
hc_new = zeros(1,length(hc));
for i = 1:length(hc)
    hc_new(i) = 2 * hc(i).* cos(wc.*(i-1));
end
ans_1c = hc_new;   % A Band-pass Filter



%%
% QUESTION 2
[n2, x2, x2mod, fs2, fs2n] = exam02_q2('12079483');

% QUESTION 1(a)
ans_2a = fft(x2);    % DFT of x[n]
w_2a = 2*pi/length(n2).*n2;
% QUESTION 1(b)
ans_2b = 0:(length(x2)-1);    % DFT indices k

% QUESTION 1(c)
ans_2c = 2*pi/length(x2)*ans_2b;    % DTFT frequencies \omega_k

% QUESTION 1(d)
ans_2d = fs2/length(x2)*ans_2b;    % CTFT frequencies f_k

% QUESTION 1(e)
if length(x2mod)<length(x2)
    result = 'downsampled';
elseif length(x2mod)>length(x2)
    result = 'upsampled';
elseif length(x2mod)==length(x2)
    result = 'none';
end    
ans_2e = result;    % Two options: 'upsampled' or 'downsampled'

% QUESTION 1(f)
if length(x2mod)<length(x2)
    factor = length(x2)/length(x2mod);
else
    factor = length(x2mod)/length(x2);
end 
ans_2f = factor;    % Upsampling or downsampling factor (a scalar number)

% QUESTION 1(g)
[num,dem] = rat(fs2n/fs2);
x1 = upsample_interp_func(x2,num);
x_new = downsample_antialias_func(x1,dem);
ans_2g = x_new;  % Resampled signal


%%
% QUESTION 3
[x3, s3, fs3] = exam02_q3('12079483');

% QUESTION 3(a)
ans_3a = length(x3)/fs3;    % Signal length, in seconds

% QUESTION 3(b)
w_3b = 1200;
[xSTFT,M1] = stft_woverlap(x3, w_3b);
[sSTFT,M2] = stft_woverlap(s3, w_3b);

figure(3);
imagesc(0:length(x3)/fs3,-pi:0.1*pi:pi,abs(xSTFT));
figure(4);
imagesc(0:length(s3)/fs3,-pi:0.1*pi:pi,abs(sSTFT));

ans_3b = 4;    % Number of reflections

% QUESTION 3(c)                     
ans_3c = [5,25,30,35];  % Time delays, in seconds (this should be an array -- order does not matter)

% QUESTION 3(d)

ii = [2.00968,2.8953,2.26645,2.76953];  % Frequency shifts, in Hz (this should be an array -- order does not matter)
io=ii*fs3/(2*pi);
ans_3d =io;


%%
% QUESTION 4
[M, G, V, fs4] = exam02_q4('12079483');

% QUESTION 4(a)
G_new = zeros(size(G));
for i = 1:M
    G_new(:,i) = flipud(G(:,i));
end

output = fb_synthesis(V,G_new);

% sound(output,fs4);
ans_4a = output;  % Reconstructed data
% QUESTION 4(b)

w_4b = 1200;
[xSTFT,M0] = stft_woverlap(ans_4a, w_4b); 
figure(5);
imagesc(0:length(output)/fs4,-pi:0.1*pi:pi,abs(xSTFT));

% x_max = max(abs(xSTFT));
% x_mean = mean(abs(xSTFT));
len_4b = length(xSTFT(:,1));
bandstop = (len_4b/3+1):len_4b *2/3;
x_mean1 = mean(abs(xSTFT(((1-1/4)*len_4b):len_4b,:)));
x_noise = max(x_mean1);
xSTFT(bandstop,:)=0;
ySTFT = xSTFT;

ySTFT(abs(xSTFT)<x_noise) = 0;
ySTFT(abs(xSTFT)>x_noise) = ySTFT(abs(xSTFT)>x_noise)-x_noise;
output_new = istft_woverlap(ySTFT,w_4b);
figure(6);
imagesc(0:length(output)/fs4,-pi:0.1*pi:pi,abs(ySTFT));


sound(output_new,fs4);

ans_4b = output_new;  % Denoised reconstructed data

% QUESTION 4(c)
ans_4c = downsample_func(ans_4a,2);  % Sped up reconstructed data

% QUESTION 4(d)
ans_4d = downsample_func(ans_4b,2);  % Denoised and sped up reconstructed data

    
    
%% 
% =========================
% SUPPORTING FUNCTIONS
% =========================

function v = wb_analysis(x,h)

   % INITIALIZE LENGTHS
    Q = length(x);      % Number of Samples
    M = size(h,2);      % Number of Filters
    N = size(h,1);      % Length of Filters
    
    % FORCE INPUT X TO BE A MULTIPLE OF FILTER LENGTH
    xtmp = zeros(ceil(Q/N)*N,1); xtmp(1:length(x)) = x; x = xtmp; Q = length(x);
    
    % INTIALIZE OUTPUTS
    v = cell(M,1);
    
    % APPLY ANALYSIS WAVELET HERE ===================
    for ii = 1:M
        ztmp = conv(x,h(:,ii));
        v{ii} = downsample_func(ztmp,2^min(ii,M-1));
    end
    % ==================================================

end

function y = wb_synthesis(v,fr)

    % INITIALIZE LENGTHS
    M = size(fr,2);           % Number of Filters
    N = size(fr,1);           % Length of Filters
    Qp = max(cellfun(@length,v))*M+N-1;     % Number of Samples
    
    % INTIALIZE OUTPUTS
    w = zeros(Qp, M);
    
    % APPLY SYNTHESIS WAVELET BANK HERE ==================
    for ii = 1:M
        rtmp = upsample_func(v{ii},2^min(ii,M-1));
        wtmp = conv(rtmp,fr(:,ii));
        w(1:length(wtmp),ii) = wtmp;
    end
    y = sum(w,2);
    % ==================================================
    
end

function v = fb_analysis(x,h)

    % INITIALIZE LENGTHS
    Q = length(x);      % Number of Samples
    M = size(h,2);      % Number of Filters
    N = size(h,1);      % Length of Filters
    
    % INTIALIZE OUTPUTS
    v = zeros(ceil((Q+N-1)/M), M);
    
    % APPLY ANALYSIS FILTERBANK HERE ===================
    for ii = 1:M
        ztmp = conv(x,h(:,ii));
        v(:,ii) = downsample_func(ztmp,M);
    end
    % ==================================================

end

function y = fb_synthesis(v,g)

    % INITIALIZE LENGTHS
    M = size(g,2);           % Number of Filters
    N = size(g,1);           % Length of Filters
    Qp = size(v,1)*M+N-1;    % Number of Samples
    
    % INTIALIZE OUTPUTS
    w = zeros(Qp, M);
    
    % APPLY SYNTHESIS FILTERBANK HERE ==================
    for ii = 1:M
        rtmp = upsample_func(v(:,ii),M);
        w(:,ii) = conv(rtmp,g(:,ii));
    end
    y = sum(w,2);
    % ==================================================
    
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

function y = upsample_interp_func(x,M)

    Q = length(x);
    y = zeros(M*Q,1);
    y(1:M:(M*Q)) = x;
    y = M*lpf_func(y,pi/M,5);

end

function y = downsample_antialias_func(x,N)

    Q = length(x);
    x = lpf_func(x,pi/N,5);
    y = x(1:N:Q);

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
        sp(k) = wc*exp(1j*(pi*(2*k-1)/(2*P)+ pi/2));
        zp(k) = exp(sp(k));
    end
    
    % CONVERT POLES AND ZEROS TO B AND A
    [bz,az] = pz2ba(zp,0);
    
    % NORMALIZE 
    bz = bz.*prod(1-zp);
    
    % APPLY FILTER
    y  = real(filter(bz,az,x)); % Real is used to remove round-off errors

end

function [xSTFT,M] = stft_woverlap(x, W)

    % CHOOSE WINDOW 
    N       = length(x);            % Number of samples
    M       = floor(N/W);           % Number of frames
    xSTFT   = zeros(W,2*M-1);       % Initialize short-time Fourier transform

    % LOOP OVER SEGMENTS
    for m = 1:(M*2-1)
        z = x((W*(m-1)/2+1):(W*(m-1)/2+W));            % Get data segment
        xSTFT(:,m) = fft(z);                           % Fourier Transform 
    end


end

function [x] = istft_woverlap(xSTFT, W)

    % CHOOSE WINDOW 
    M       = (size(xSTFT,2)+1)/2;     % Number of frames
    N       = M*W;                     % Number of samples
    x       = zeros(N,1);
    
    % LOOP OVER SEGMENTS
    for m = 1:(M*2-1)
        x((W*(m-1)/2+1):(W*(m-1)/2+W)) = x((W*(m-1)/2+1):(W*(m-1)/2+W)) + real(ifft(xSTFT(:,m))).*hann(W); % Inverse Fourier Transform
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

function w = hann(N)

    w = 0.5 - 0.5 .* cos (2 .* pi .* (0 : (N-1))' ./ (N-1));
    w = w(:); 

end
