import matplotlib.pyplot as plt
import numpy as np

N = 100000
#generate random array elements of size N = 100000
h = (1/np.sqrt(2))*(np.random.randn(N) + np.random.randn(N)*1j)

#get absolute value of elements in array a
a = [np.abs(h[i])  for i in range(1,N)]

#get phase of elements in array phi
phi = [np.angle(h[i])for i in range(1,N)]

#get histogramic distribution of array a and array phi
pdfa, bin_edges = np.histogram(a,np.arange(0,4.05,0.05))
pdfp, bin_edges = np.histogram(phi,np.arange(-np.pi,np.pi,0.05))


#get plot of modulus graph
fig1 = plt.figure()
ax = fig1.add_axes([0,0,1,1])
ax.bar(np.arange(0,4,0.05), [ele/N/0.05 for ele in pdfa])
ax.set_title('PDF of Amplitude')

#get plot phase graph
fig2 = plt.figure()
ax1 = fig2.add_axes([0,0,1,1])
ax1.bar(np.arange(-np.pi,np.pi-0.05,0.05), [ele/N/0.05 for ele in pdfp])
ax1.set_title('PDF of Phase')


#show plots
plt.show()