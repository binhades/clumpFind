from astropy.io import fits
from astrodendro import Dendrogram

hdu = fits.open('orion_12CO.fits')[0]
data = hdu.data
hd = hdu.header

print(data)
d = Dendrogram.compute(data)


