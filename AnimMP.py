import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mgimg
from matplotlib import animation
from moviepy.editor import *
import glob

files = []
incl = 35
Omega = 312


Foldername = 'ecc_incl'+str(incl)+'_Omega'+str(Omega)
files = np.sort(glob.glob(Foldername+'\*.png'))

print(files)
clip = ImageSequenceClip(Foldername+'/', fps=24) 
clip.write_videofile(Foldername+".mp4", fps = 24)
clip.write_gif(Foldername+'.gif')
