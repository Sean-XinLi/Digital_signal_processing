%% Question 1

clear;

% -----------------
% QUESTION 1 SETUP
% -----------------
M = 10;   % Number of channels

% AUDIO FILE
[x, fs] = audioread('zoqfotpik.wav');

% -------------------
% QUESTION 1(a) -- function template at the end of the file
% -------------------

% -------------------
% QUESTION 1(b)
% -------------------
ans_h = [1/sqrt(2) -1/sqrt(2)];   %INSERT ANSWER HERE% 
ans_g = [1/sqrt(2) 1/sqrt(2)];   %INSERT ANSWER HERE%
h=zeros(512,9);
for i=1:M-1
    h(1:length(upsample_func(ans_h,2^(i-1))),i)=upsample_func(ans_h,2^(i-1));
end
g=zeros(512,9);
g1=ans_g';
g(1:length(g1),1)=g1;
g2=conv(g1,upsample_func(ans_g,2^1));
g(1:length(g2),2)=g2;
g3=conv(g2,upsample_func(ans_g,2^2));
g(1:length(g3),3)=g3;
g4=conv(g3,upsample_func(ans_g,2^3));
g(1:length(g4),4)=g4;
g5=conv(g4,upsample_func(ans_g,2^4));
g(1:length(g5),5)=g5;
g6=conv(g5,upsample_func(ans_g,2^5));
g(1:length(g6),6)=g6;
g7=conv(g6,upsample_func(ans_g,2^6));
g(1:length(g7),7)=g7;
g8=conv(g7,upsample_func(ans_g,2^7));
g(1:length(g8),8)=g8;
g9=conv(g8,upsample_func(ans_g,2^8));
g(1:length(g9),9)=g9;
g(513:1014,:)=[];

y=zeros(512,10);
for i =1:8
    y(1:length(conv(h(:,i+1),g(:,i))),i+1)=conv(h(:,i+1),g(:,i));
end
y(1:length(h(:,1)),1)=h(:,1);

y(1:length(h(:,1)),10)=g(:,9);
y(513:1023,:)=[];

% ans_f=wb_filters(ans_h,ans_g,M);
ans_f = y;

% -------------------
% QUESTION 1(c) -- function template at the end of the file
% -------------------


% -------------------
% QUESTION 1(d) -- function template at the end of the file
% -------------------


% -------------------
% QUESTION 1(e)
% -------------------
ans_v = wb_analysis(x,ans_f);

figure();
set(gcf, 'Units', 'Inches', 'Position', [1 0 3 7])
for m = 1:M
subplot(M,1,m)
plot(ans_v{m}); 
ylabel(['#' num2str(m)])
xlim([0 length(x)/2^m])
end
xlabel(['Samples'])

% -------------------
% QUESTION 1(f)
% -------------------
% PERFORM INSTRUCTIONS HERE
hy = ans_v;
hy(1,1) = {[]};
for i= 3:10
    hy(i,1)={[]};
end
ans_vr = hy

% -------------------
% QUESTION 1(g)
% -------------------
ans_fr = flipud(ans_f);   %INSERT ANSWER HERE% 
ans_y = wb_synthesis(ans_vr,ans_fr);

tx = 1/fs:1/fs:length(x)/fs;
ty = 1/fs:1/fs:length(ans_y)/fs;

figure();
subplot(2,1,1)
plot(tx,x); 
xlabel(['Time [s]'])
title('Original signal')
subplot(2,1,2)
plot(ty,ans_y); 
title('Reconstructed signal')
xlabel(['Time [s]'])
xlim([2^(M-1)/fs 2^(M-1)/fs+length(x)/fs])

% disp('Playing original audio')
% soundsc(x,fs)
% pause(length(x)/fs)
% 
% disp('Playing modified audio')
% soundsc(ans_y,fs)


%% 
% ========================= SUPPORTING FUNCTIONS =========================

function f = wb_filters(h,g,M)
    
    % INTIALIZE OUTPUTS
    P = length(h); assert(length(h) == length(g));
    f = zeros((P-1)*2^(M-1)+(P-1)-1, M);
    h = zeros(2^(M-1),M-1);
    g = zeros(2^(M-1),M-1);
    % OBTAIN FILTERS HERE ===================
    for ii = 1:M
        for hi =1:M-1
            h(1:length(upsample_func(ans_h,2^(hi-1))),hi)=upsample_func(h,2^(hi-1));
        end
        for yi = 2:M
            g(1:P,yi)=conv(g,upsample_func(g,2^(yi-1)));
        end
        g(513:1014,:)=[];
        f(1:length(conv(h(:,ii-1),g(:,ii-1))),ii-1)=conv(h(:,ii-1),g(:,ii-1));
        f(1:length(h(:,1)),1)=h(:,1);
        f(1:length(h(:,1)),10)=g(:,9);
        f(513:1023,:)=[];
    end
            
% %         yy=0:P;
% %         h(yy+1,ii+1)=yy+2^(2^(ii-1));
% %         g(yy+1,ii+1)=conv(g,yy+2^(2^(ii-2)));
% %         
%         
%         % OBTAIN FILTER FOR EACH CHANNEL
%         f(:,ii) = conv(aa,hy);
%     end 
%     % ==================================================

end

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
    for ii = 1:M-1
            
        ztmp = conv(x,h(:,ii));
       
  
        % OBTAIN COFFICIENTS FOR EACH CHANNEL
        v(ii,1) = {downsample_func(ztmp,2^ii)};
    end
    % ==================================================
        ztmp=conv(x,h(:,10));
        v(10,1)={downsample_func(ztmp,2^9)};
end


function y = wb_synthesis(v,fr)

    % INITIALIZE LENGTHS
    M = size(fr,2);           % Number of Filters
    N = size(fr,1);           % Length of Filters
    Qp = max(cellfun(@length,v))*M+N-1;     % Number of Samples
    
    % INTIALIZE OUTPUTS
    w = zeros(Qp, M);
    
    % APPLY SYNTHESIS WAVELET BANK HERE ==================
    l=zeros(68864,10);
    for i =1:10
        l(1:length(cell2mat(v(i,1))),i)=cell2mat(v(i,1));
    end
    for ii = 1:M-1
         rtmp = upsample_func(l(:,ii),2^ii);
        w(1:length(conv(rtmp,fr(:,ii))),ii) = conv(rtmp,fr(:,ii));% OBTAIN COFFICIENTS FOR EACH CHANNEL
      
    end
    rtmp=upsample_func(l(:,10),2^9);
    w(1:length(conv(rtmp,fr(:,10))),10) = conv(rtmp,fr(:,10));
    y = sum(w,2);  % Sum over channels
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


function y = upsample_func_trun(x,M)

    Q = length(x);
    y = zeros(M*Q-(M-1),1);
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