import numpy as np
from astropy import units as u
#import MCMC
import os
import matplotlib.pylab as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
import matplotlib.ticker as ticker
# from matplotlib.collections import PatchCollection
# import matplotlib.patches as mpatches
# import time
from scipy.signal import find_peaks

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ============================================================================
"   Plots the geometry of the Roche equipotential
    ============================================================================
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
================================
Binary system and jet properties
================================
"""
km_to_au = 1/1.495979E8
cm_to_Rsol = 1/6.95700E11
m_to_au = 1.495979E11
sec_to_year = 1/(3600*24*365.15)
kg_to_Msol =1.98847E30
Msol_per_yr_to_g_s = 6.307394723489999E27

AU_to_km        = 1.496e+08     # 1AU in km
days_to_sec     = 24*60*60      # 1day in seconds
degr_to_rad     = np.pi/180.    # Degrees to radians
G           = 6.674e-11


sec_in_yr = 60*60*24*365
Rsol_in_au = 215.032
e = np.exp(1)
fparam = 0


jet = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

def lobe_surf(xfit, yfit):
    u = np.linspace(0, 2 * np.pi, 100)
    xl = np.outer(xfit,np.ones(np.size(u)) )
    yl = np.outer(yfit, np.cos(u))
    zl = np.outer(yfit, np.sin(u))

    return xl, yl, zl
def sphere(pos, r):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = pos[0] + r * np.outer(np.cos(u), np.sin(v))
    y = pos[1] + r * np.outer(np.sin(u), np.sin(v))
    z = pos[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))

    return x,y,z

def companion_mass(f, m1, ind):
    p = []
    i = ind*np.pi/180
    p.append(-np.sin(i)**3)
    p.append(f)
    p.append(2*m1*f)
    p.append(f*m1**2)
    r =np.roots(p)
    real_valued = r.real[abs(r.imag)<1e-5]
    M= real_valued
    M2 = M[0]
    print("inclinaison [°] = "+  str(ind)+ ' companion mass = '+ str(round(M[0],2)))
    return M2


def circle(radius, x0 = 0, y0 = 0, theta_value = np.inf): #radisu in mas
    radius = radius
    theta = np.linspace( 0 , 2 * np.pi , 180 )
    if theta_value != np.inf:
        theta = theta_value
    x = radius * np.cos( theta ) +x0
    y = radius * np.sin( theta )+ y0
    return x, y

def potential(x, y, prim_pos, sec_pos, prim_mass, sec_mass, sma_AU):
    r = (x**2 + y**2)**0.5
    s1 = ((x-prim_pos[0])**2 + (y-prim_pos[1])**2)**0.5
    s2 = ((x-sec_pos[0])**2 + (y-sec_pos[1])**2)**0.5
    G = np.pi*4*np.pi
    pot = - G*(((1-fparam)*prim_mass/s1 + sec_mass/s2) + 0.5*((prim_mass + sec_mass)*(r**2)/(sma_AU**3)))
    pot[pot <-2085] = -2085
    return pot

def rotation_tot(inclination, omega, x,y,z):
    x,y = rotation(omega-90, x,y)
    y, z = rotation(inclination, y, z)
    #z, y = rotation(omega, z, y)
    return x,y,z

def deriv_pot(x, y, prim_pos, sec_pos, prim_mass, sec_mass, sma_AU):
    #gradient of potentiel in the orbital plane
    d1 =((x-prim_pos[0])**2 + (y-prim_pos[1])**2)**0.5
    d2 =((x-sec_pos[0])**2 + (y-sec_pos[1])**2)**0.5
    G = np.pi*4*np.pi
    dx = -G*((1-fparam)*prim_mass*(x -prim_pos[0])/d1**3 +sec_mass*(x -sec_pos[0])/d2**3 - (prim_mass + sec_mass)*x/(sma_AU**3))
    dy = -G*((1-fparam)*prim_mass*(y -prim_pos[1])/d1**3 +sec_mass*(y -sec_pos[1])/d2**3 - (prim_mass + sec_mass)*y/(sma_AU**3))
    return (dx, dy)



def rotation_new(inclination, omega, x,y,z):
    omega = omega*np.pi/180 
    i = - inclination*np.pi/180
    ux = np.cos(omega - np.pi/2)
    uy = np.sin(omega - np.pi/2)
    uz = 0
    xprim = (np.cos(i) + ux*ux*(1-np.cos(i)))*x + (ux*uy*(1-np.cos(i)) - uz*np.sin(i))*y + (ux*uz*(1-np.cos(i)) + uy*np.sin(i))*z
    yprim = (uy*ux*(1-np.cos(i)) + uz*np.sin(i))*x + (np.cos(i) + uy*uy*(1-np.cos(i)))*y + (uy*uz*(1-np.cos(i)) - ux*np.sin(i))*z
    zprim = (ux*uz*(1-np.cos(i)) - uy*np.sin(i))*x + (uz*uy*(1-np.cos(i)) + ux*np.sin(i))*y + (np.cos(i) + uz*uz*(1- np.cos(i)))*z
    return xprim, yprim, zprim

def lagrange(x, prim_pos, sec_pos, prim_mass, sec_mass, sma_AU):
    a3 = sma_AU**3
    y1 = prim_pos[1]
    x1 = prim_pos[0]
    y2 = sec_pos[1]
    x2 = sec_pos[0]
    M1 = prim_mass
    M2 = sec_mass
    y = 0
    s1 = ((x-x1)**2 + (y-y1)**2)**0.5
    s2 = ((x-x2)**2 + (y-y2)**2)**0.5
    r = (x**2 + y**2)**0.5
    sol = - np.pi*np.pi*4*(((1-fparam)*prim_mass/s1 + sec_mass/s2) + 0.5*((prim_mass + sec_mass)*(r**2)/(sma_AU**3)))
    return sol


    
def ring(radius):
    n = 36
    nj = n*1j
    u, v = np.mgrid[0:2*np.pi:nj, 0.49*np.pi:0.51*np.pi:1j]
    x = radius*np.cos(u)*np.sin(v)
    y = radius*np.sin(u)*np.sin(v)
    z = radius*np.cos(v)
    return x, y, z

def equipotential(incl=35, plot = False, plot3d = False, plotRot = False, colorplot = 'k'):

    period              = 11.0*365.25
    incl_rad            = incl*np.pi/180
    sma_AU              = 7.05
    #asini               = 0.5
    #fm                  = 0.323

    primary_mass            = 1.5
    secondary_mass          = 0.9

    mass_ratio              = primary_mass / secondary_mass
    print('Giant mass : ', primary_mass)
    print('mass ratio = ', mass_ratio)
    primary_sma_AU          = sma_AU/(1+ mass_ratio)

    secondary_sma_AU        = mass_ratio * primary_sma_AU

    print('a = ', sma_AU)
    b1                      = (0.5 -0.227*np.log10(mass_ratio))*sma_AU

    prim_pos                = (primary_sma_AU,0,0)
    sec_pos                 = (-secondary_sma_AU,0,0)
    print(prim_pos, sec_pos)



    b1_pos                  = (-secondary_sma_AU +b1, 0, 0)
    ndata = 2000
    lengthMAX = 8.05
    lengthMIN = -8.05

    dimx = (lengthMIN,lengthMAX)
    dimy = (lengthMIN,lengthMAX)
    dimz = (lengthMIN,lengthMAX)
    x = np.linspace(dimx[0], dimx[1], ndata)
    y = np.linspace(dimy[0], dimy[1], ndata)
    z = np.linspace(dimz[0], dimz[1], ndata)
    X, Y= np.meshgrid(x, y)
    equipot = potential(X,Y, prim_pos, sec_pos, primary_mass, secondary_mass, sma_AU)
    dPx, dPy= deriv_pot(X,Y, prim_pos, sec_pos, primary_mass, secondary_mass, sma_AU)
    norm_dP = (dPx**2 + dPy**2)**0.5
    norm_dP[norm_dP >1] = 1

    pic = find_peaks(lagrange(x,prim_pos, sec_pos, primary_mass, secondary_mass, sma_AU))
    pic = pic[0]

    levels = np.sort(lagrange(x[pic],prim_pos, sec_pos, primary_mass, secondary_mass, sma_AU))
    print('level = ',levels)
    level = levels[0]
    print('level')
    print('Lagrange point position ', x[pic])
    dL1 = prim_pos[1]+ sma_AU*(0.5 -0.227*np.log10(secondary_mass/primary_mass)) #distance to L1
    period_yr = period/365.25
    # yvec = xcoord, ycoord, vxcoord, vycoord
    xL1 = b1 + sec_pos[0]
    print(xL1)
    # print(stop)


    # plot = True
    if plot:

        plt.imshow(equipot*0, extent = (lengthMIN, lengthMAX, lengthMIN, lengthMAX),origin = 'lower',cmap='Greys')#, vmin = -50, vmax = 0)


        #xL, yL = circle((0.709/2)/1.09,x0 = prim_pos[0], y0 = prim_pos[1])
        # plt.plot(xL,yL, '-r')
        Cp=plt.gca().contour(x, y, equipot, levels, colors=['r', 'k', 'k'])
        # for i in range(0,len(pic)):
        #     plt.plot(y[pic[i]]*np.cos(np.arctan2(prim_pos[1],prim_pos[0])), y[pic[i]]*np.sin(np.arctan2(prim_pos[1],prim_pos[0])), 'ok')
        # plt.colorbar()
        plt.xlabel('X [AU]')
        plt.ylabel('Y [AU]')
        limm = 10.1
        plt.xlim([-limm,limm])
        plt.ylim([-limm,limm])
        # Ct=plt.contour.ContourSet(x, y, equipot, level)
        plt.plot(prim_pos[0], prim_pos[1], 'ok')
        plt.plot(sec_pos[0], sec_pos[1], 'xk')
        # plt.plot(b1_pos[0], b1_pos[1], 'og')
        print(b1_pos)

        ax = plt.gca()
        plt.minorticks_on()
        ax.tick_params(direction="in")
        ax.tick_params(direction="in",which = 'both', top=True, right=True )
        plt.savefig('Roche.png',   bbox_inches='tight')
        #plt.savefig('Roche.pdf',   bbox_inches='tight')
        plt.show()

                # surf = ax.plot_surface(X, Y, equipot,linewidth=1,cmap = 'Reds', antialiased=False)
        if plot3d:
            phase = np.linspace(0,1, 1)
            phase = np.linspace(0,1, 41)
            for phi in phase:
                print(phi)
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize = (10,10))

                # Cp=plt.gca().contour(X, Y, equipot, level, colors=['k', 'k', 'k'])
                mask = (np.abs(equipot - level)<0.01)
                mask = np.logical_and(mask, Y >=0)
                # Xlin = np.linspace(np.min(X[mask]))
                array = [X[mask], Y[mask]]
                Xn = X[mask][np.argsort(X[mask])]
                Yn = Y[mask][np.argsort(X[mask])]
                xfit = np.linspace(np.min(Xn), np.max(Xn), 500)
                yfit = np.interp(xfit, Xn, Yn)
                c = np.polyfit(xfit, yfit, 10)
                p = np.poly1d(c)
                demark = 0.25
                #xl, yl, zl =  lobe_surf(xfit[xfit>demark], yfit[xfit>demark])
                #xlc, ylc, zlc =  lobe_surf(xfit[xfit<=demark], yfit[xfit<=demark])
                xlt, ylt, zlt =  lobe_surf(xfit, yfit)
                #print(np.shape(xlt), np.shape(ylt), np.shape(zlt))

                Rprim = 0.2
                x_R, y_R, z_R = sphere(prim_pos, Rprim)
                #ax.plot_surface(x_R, y_R, z_R, color = 'red', antialiased=False)

                ax = plt.gca()
                ax.plot_surface(xlt-prim_pos[0], ylt, zlt,  alpha=0.1, shade= False, antialiased=False)
                plt.plot(xfit-prim_pos[0], yfit, color = 'k', lw = 3)
                plt.plot(xfit-prim_pos[0], -yfit, color = 'k', lw = 3)
                #ax.plot_surface(xl, yl, zl, color = 'orangered', alpha = 0.5)
                #ax.plot_surface(xlc, ylc, zlc, color = 'goldenrod', alpha = 0.5)#,antialiased=False)


                #plt.plot(overflow[0], overflow[1], ls = '-', color = 'goldenrod', linewidth = 1)


                # plt.plot(xcir,ycir, ':b')
                jet_lenght = 1.2
                # plt.plot(xtidal,ytidal, '-.b')
                off = 0.1

                lim = 10.1
                ax.set_xlim([-lim,lim])
                ax.set_ylim([-lim,lim])
                ax.set_zlim([-lim,lim])
                # plt.plot(xfit, yfit, '-')
                elev = 90 - incl
                #elev = 90
                # phi = 1
                azim = 360*phi
                Omega = 312 #312
                #omega = Omega
                roll = Omega-90
                #plt.title('phase = ' +str(np.round(phi,2))+r', $i$ = ' +str(int(incl)) + '°')
                ax.view_init(elev, -azim, roll)
                plt.plot(prim_pos[0]-prim_pos[0], prim_pos[1], 'or')
                plt.plot(sec_pos[0]-prim_pos[0], sec_pos[1], 'ok')

                print(np.shape(Xn))

                ax.set_xlabel('X [AU]')

                #ax.arrow(0, 0, 1, 1)

                ax.set_ylabel('Y [AU]')
                ax.set_zlabel('Z [AU]')
                ax.set_aspect('equal')
                tikx = [-1, -0.5, 0, 0.5, 1]
                #ax.set_xticks(tikx)
                #ax.set_yticks(tikx)
                #ax.set_zticks(tikx)
                ax.set_axis_off()


                OutputDir = 'incl'+str(int(incl))+'_Omega'+str(int(Omega))
                if not os.path.exists(OutputDir):
                    os.makedirs(OutputDir)
                plt.savefig(OutputDir+'/{:003.0f}'.format(phi*100) + '.png',   bbox_inches='tight')
                #plt.savefig(str(round(phi*100)) + '.pdf',   bbox_inches='tight')
                plt.close()
                #plt.show()

        if plotRot:
            inclination = incl

            plx = 6.1
            xeq,yeq,zeq = rotation_new(inclination, omega, xlt, ylt, zlt)

            ax = plt.figure().add_subplot(projection='3d')
            #plt.title('Phase = '+str(round(phase,2)))
            ax.plot(xeq*plx, yeq*plx, zeq*plx, '--', label='Secondary')
            ax.view_init(90, 270)
            ax.set_aspect('equal')
            ax.set_xlabel('x [mas]')
            ax.set_ylabel('y (N->) [mas]')
            ax.set_zlabel('z [mas]')
            plt.show()



    #
# #
# #
equipotential(incl=35,plot = True, plot3d = True, plotRot = False)
# equipotential(incl=60, primary_mass=1,plot = False, plot3d = False)
# equipotential(incl=70, primary_mass=1,plot = False, plot3d = False)
# # equipotential(incl=60, primary_mass=1.1,plot = False, plot3d = False) #not possible because Mwd> 1.5
# equipotential(incl=70, primary_mass=1.1,plot = False, plot3d = False)
# equipotential(incl=60, primary_mass=0.9,plot = False, plot3d = False)
# equipotential(incl=70, primary_mass=0.9,plot = False, plot3d = False) #unprobable because because Mwd= 1.2


plt.show()

