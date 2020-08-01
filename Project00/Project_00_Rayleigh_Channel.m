% Rayleigh fading channel
close all;
clear all;

N = 10000000;
h = 1/sqrt(2)*(randn(1,N)+1j*randn(1,N));
a = abs(h);
phi = angle(h);

pdfa=hist(a,[0:0.05:4]);
bar([0:0.05:4],pdfa/N/0.05)
title('PDF of Amplitude')

pdfp=hist(phi,[-pi:0.05:pi]);
figure;
bar([-pi:0.05:pi],pdfp/N/0.05)
title('PDF of Phase')