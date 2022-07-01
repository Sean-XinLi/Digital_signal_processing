function y = sawtooth(t,width)

    t0 = t / (2*pi);
    y = 2*(t0-floor(t0))-1;

end
