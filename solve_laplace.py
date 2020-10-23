# tested for python2.7.18 and python3.7.4 
# vtk functionality doesn't work for python3

import os
import sys
import numpy as np
from discrete_differentials import secondir_matrix
from scipy.special import legendre
from scipy.optimize import curve_fit
import scipy.sparse as ssp
import scipy.sparse.linalg

import resource
soft, hard = 10*2**30, 10*2**30
resource.setrlimit(resource.RLIMIT_AS,(soft, hard))

base = '/home/gary/azimuthal_dielectrics/'


def Pol(x, l):
    coeffs = legendre(l)
    ret = np.zeros_like(x)
    for n in range(l+1):
        ret = coeffs[l-n]*x**n
    return ret


def matrix_tostring(M, tag):
    if not isinstance(M, np.ndarray): M = M.toarray()

    st = ''
    for head in tag:
        head = str(head)
        if len(head) <= 8:
            st = st + (8-len(head))*' ' + head
        else:
            st = st + head[:8]
    st = st + '\n'
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            st = st + '%8.3f'%(M[i,j])
            #st = st + '  '
        st = st + '  |  ' + str(tag[i])
        st = st + '\n'
    return st

def mv_tostring(M, v, tag):
    if not isinstance(M, np.ndarray): M = M.toarray()
    if not isinstance(v, np.ndarray): v = v.toarray()

    st = ''
    for head in tag:
        head = str(head)
        if len(head) <= 8:
            st = st + (8-len(head))*' ' + head
        else:
            st = st + head[:8]
    st = st + '\n'
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            st = st + '%8.3f'%(M[i,j])
            #st = st + '  '

        st = st + '  |  ' + str(tag[i])
        st = st + '  | %8.3f'%(v[i])

        st = st + '\n'

    return st

def vect_tostring(v, tag):
    if not isinstance(v, np.ndarray): v = v.toarray()

    st = ''
    for i in range(v.size):
        st = st + '%8.3f'%(v[i]) + '  |  ' + str(tag[i]) + '\n'
    return st

#A = np.arange(9).reshape((3,3))
#x = 10*np.arange(3)
#print(mv_tostring(A,x))
#print(matrix_tostring(A,list(x)))


def sigma(n, N=-1, M=-1):
    assert(isinstance(n,int))
    assert(0 <= n <= N*M-1)
    return (n//M, n%M) 

def invsigma(i, j, N=-1, M=-1):
    assert(isinstance(i,int))
    assert(isinstance(j,int))
    assert(0 <= i <= N-1)
    assert(0 <= j <= M-1)
    return M*i + j

def r2u(r):
    return -1./r

def u2r(u):
    return -1./u

def theta2eta(theta):
    if isinstance(theta, np.ndarray):
        shape = theta.shape
        theta = theta.flatten()
        eta = np.zeros(theta.size)
        for i in range(theta.size):
            eta[i] = theta2eta(theta[i])
        return eta.reshape(shape)

    #eta = np.log(np.sin(theta)/(1.+np.cos(theta))) original
    c = np.cos(theta)
    x = (1.-c)/(1.+c)
    eta = 0.5 * np.log(x)
    return eta

def eta2theta(eta):
    if isinstance(eta, np.ndarray):
        shape = eta.shape
        eta = eta.flatten()
        theta = np.zeros(eta.size)
        for i in range(eta.size):
            theta[i] = eta2theta(eta[i])
        return theta.reshape(shape)

    x = np.exp(2.*eta)
    c = (1.-x)/(1.+x)  # or c = -np.tanh(eta)
    theta = np.arccos(c)
    #theta = np.arccos( (-x + np.sqrt(2.*x**2 - 2.*x + 1))/(1-x) ) # doesnt work? arithmetic mistake?
    return theta


def diamatmul(v, M):
    #sparse equivalent to  np.diag(v) @ M     ( @ is np.matmul(,) in python3 )

    #https://stackoverflow.com/questions/12237954/multiplying-elements-in-a-sparse-array-with-rows-in-matrix
    # just to make the point that this works only with CSR:
    if not isinstance(M, scipy.sparse.csr_matrix):
        raise ValueError('Matrix must be CSR (scipy.sparse.csr_matrix)')

    Z = M.copy()
    # simply repeat each value in Y by the number of nnz elements in each row: 
    Z.data *= v.repeat(np.diff(Z.indptr))

    return Z


def solve_noDielectric(r_min, r_max, phi0, N=100, sig=9., debug=False):
    u_min = r2u(r_min)
    u_max = r2u(r_max)

    deta = 2.*sig/(N+1)
    du = (u_max-u_min)/(N+1)

    u_ax = np.linspace(u_min + du, u_max - du, N)
    eta_ax = np.linspace(-sig + deta, sig - deta, N)

    u_op = diamatmul( u_ax**2 , secondir_matrix(N, bc='dirichlet')/du**2 )

    rx = np.exp(eta_ax)
    eta_op = diamatmul( ((1.+rx**2)/(2.*rx))**2 , secondir_matrix(N, bc='neumann')/deta**2 )

    scaledlap = ssp.kron(ssp.identity(N,format='csr'), eta_op) + ssp.kron(u_op, ssp.identity(N,format='csr'))

    if debug:
        print('N, sig = %d, %f \n'%(N,sig))
        print('phi0 = %f'%(phi0))
        print('r_min = %f'%(r_min))
        print('r_max = %f'%(r_max))
        print('u_min = %f'%(u_min))
        print('u_max = %f\n'%(u_max))

    innerbc = -( phi0*(r_min**2)/du**2 )*np.ones(N)
    e0 = np.zeros(N)
    e0[0] = 1. 
    inhomo = np.kron(e0, innerbc)

    if debug: 
        #print(inhomo)
        #print(scaledlap)
        print(mv_tostring(scaledlap.toarray(), inhomo, [sigma(m, N=N, M=N) for m in range(N**2)]))

    result = scipy.sparse.linalg.spsolve(scaledlap, inhomo)
    if debug:
        print('\n')
        print(vect_tostring(result, [sigma(m, N=N, M=N) for m in range(N**2)]))
        print('\n')
    result = result.reshape((N,N))

    r_ax = u2r(u_ax)
    theta_ax = eta2theta(eta_ax)

    return [r_ax, theta_ax, result]


def solve(r_min, r_max, r_crit, phi0, epsilonU, epsilonL, N=101, sig=9., debug=False):
    assert( N%2 == 1)

    u_min = r2u(r_min)
    u_max = r2u(r_max)

    deta = 2.*sig/(N+1)
    du = (u_max-u_min)/(N+1)

    u_ax = np.linspace(u_min + du, u_max - du, N)
    eta_ax = np.linspace(-sig + deta, sig - deta, N)

    u_op = diamatmul( u_ax**2 , secondir_matrix(N, bc='dirichlet')/du**2 )

    rx = np.exp(eta_ax)
    eta_op = diamatmul( ((1.+rx**2)/(2.*rx))**2 , secondir_matrix(N, bc='neumann')/deta**2 )

    scaledlap = ssp.kron(ssp.identity(N,format='csr'), eta_op) + ssp.kron(u_op, ssp.identity(N,format='csr'))
    # kron output bsr_matrix, but cannot assign [n,m] values in that form
    # change back to csr_matrix
    scaledlap = ssp.csr_matrix(scaledlap)

    if debug:
        print('N, sig = %d, %f \n\n'%(N,sig))
        print('phi0 = %f'%(phi0))
        print('r_min = %f'%(r_min))
        print('r_max = %f'%(r_max))
        print('r_crit = %f'%(r_crit))
        print('epsilonU = %f'%(epsilonU))
        print('epsilonL = %f'%(epsilonL))
        print('u_min = %f'%(u_min))
        print('u_max = %f'%(u_max))
        print('du = %f'%(du))
        print('deta = %f\n'%(deta))

    u_crit = r2u(r_crit)
    i_crit = int(round((u_crit-u_min)/du)) - 1

    #n0 = sl.invsigma(0, (N-1)//2, N=N, M=N)
    #ns = n0 + np.arange(0,N**2,N)
    for i in range(N):
        if debug:
            print('i=%d'%(i))

        if i < i_crit:
            continue
        n = invsigma(i, (N-1)//2, N=N, M=N)
        n_oneless = invsigma(i, (N-1)//2-1, N=N, M=N) # here more/less refers to adjusting j
        n_onemore = invsigma(i, (N-1)//2+1, N=N, M=N)

        if debug:
            print('n=%d'%(n))
            #print('before: scaledlap[n, :] = ' + str(scaledlap[n, :].tolist()))

        #################### scaledlap[n, :] = 0. ######################  # see ./stackoverflow.py
        row_idx = n

        row_indices = scaledlap.indices[scaledlap.indptr[n]:scaledlap.indptr[n+1]]
        new_row_data = np.zeros_like(row_indices)

        N_elements_new_row = len(new_row_data)
        assert(N_elements_new_row == len(row_indices))

        idx_start_row = scaledlap.indptr[row_idx]
        idx_end_row = scaledlap.indptr[row_idx + 1]

        scaledlap.data = np.r_[scaledlap.data[:idx_start_row], new_row_data, scaledlap.data[idx_end_row:]]
        scaledlap.indices = np.r_[scaledlap.indices[:idx_start_row], row_indices, scaledlap.indices[idx_end_row:]]
        scaledlap.indptr = np.r_[scaledlap.indptr[:row_idx + 1], scaledlap.indptr[(row_idx + 1):]]
        ###########################################################

        scaledlap[n, n_oneless] = epsilonU/deta # will automatically print SparseEfficiencyWarning if this isn't already ocupied value
        scaledlap[n, n_onemore] = epsilonL/deta # * note in convention higher j pushes eta into the L region
        scaledlap[n, n] = -(epsilonL+epsilonU)/deta
        if debug:
            print('n_oneless=%d'%(n_oneless))
            print('n_onemore=%d'%(n_onemore))
            #print('after: scaledlap[n, :] = ' + str(scaledlap[n, :].tolist()))

    if i_crit > 0: # != -1 ?
        for j in range(N):
            # epsilon inner is just 1.

            n = invsigma(i_crit, j, N=N, M=N)
            n_oneless = invsigma(i_crit-1, j, N=N, M=N)
            n_onemore = invsigma(i_crit+1, j, N=N, M=N) # here more/less refers to adjusting i (NOT j)

            #################### scaledlap[n, :] = 0. ######################
            row_idx = n

            row_indices = scaledlap.indices[scaledlap.indptr[n]:scaledlap.indptr[n+1]]
            new_row_data = np.zeros_like(row_indices)

            N_elements_new_row = len(new_row_data)
            assert(N_elements_new_row == len(row_indices))

            idx_start_row = scaledlap.indptr[row_idx]
            idx_end_row = scaledlap.indptr[row_idx + 1]

            scaledlap.data = np.r_[scaledlap.data[:idx_start_row], new_row_data, scaledlap.data[idx_end_row:]]
            scaledlap.indices = np.r_[scaledlap.indices[:idx_start_row], row_indices, scaledlap.indices[idx_end_row:]]
            scaledlap.indptr = np.r_[scaledlap.indptr[:row_idx + 1], scaledlap.indptr[(row_idx + 1):]]
            ###########################################################

            if j > (N-1)//2: # higher j in the L region by *
                scaledlap[n, n_oneless] = 1./deta # will automatically print SparseEfficiencyWarning if this isn't already ocupied value
                scaledlap[n, n_onemore] = epsilonL/deta
                scaledlap[n, n] = -(epsilonL+1.)/deta

            elif j < (N-1)//2:
                scaledlap[n, n_oneless] = 1./deta
                scaledlap[n, n_onemore] = epsilonU/deta
                scaledlap[n, n] = -(epsilonU+1.)/deta

            else: # j == (N-1)//2
                eps_average = (epsilonU + epsilonL)/2.
                scaledlap[n, n_oneless] = 1./deta
                scaledlap[n, n_onemore] = eps_average/deta
                scaledlap[n, n] = -(eps_average+1.)/deta


    innerbc = -( phi0*(r_min**2)/du**2 )*np.ones(N)
    e0 = np.zeros(N)
    e0[0] = 1. 
    inhomo = np.kron(e0, innerbc)
    if i_crit <= 0:
        inhomo[invsigma(0, (N-1)//2, N=N, M=N)] = 0.
    else:
        assert(inhomo[invsigma(i_crit, (N-1)//2, N=N, M=N)] == 0.)

    if debug:
        print(mv_tostring(scaledlap, inhomo, [sigma(m, N=N, M=N) for m in range(N**2)]))

    result = scipy.sparse.linalg.spsolve(scaledlap, inhomo)
    if debug:
        print('\n')
        print(vect_tostring(result, [sigma(m, N=N, M=N) for m in range(N**2)]))
        print('\n')
    result = result.reshape((N,N))
    if debug:
        pass
        #print(result)

    r_ax = u2r(u_ax)
    theta_ax = eta2theta(eta_ax)

    return [r_ax, theta_ax, result]


def colorplot(r_min, r_max, r_crit, phi0, epsilonU, epsilonL, N=101, sig=9.):
    r_ax, theta_ax, result = solve(r_min, r_max, r_crit, phi0, epsilonU, epsilonL, N=N, sig=sig)

    r = np.kron(r_ax, np.ones(N)).reshape((N,N))
    theta = np.kron(np.ones(N), theta_ax).reshape((N,N))

    #print(r_ax)
    #print(1./r_ax)
    #print(theta_ax)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    axes = fig.gca()

    Nbt = 4 # Approximate number of colors between ticks for linear scale
    import matplotlib
    cmap = matplotlib.pyplot.get_cmap('viridis', Nbt*(1000-1))

    axes.set(title='title')
    axes.set(xlabel='r')
    axes.set(ylabel='theta')
    axes.axis('square')
    axes.set_xlim(r_min-0.5, r_max+0.5)
    axes.set_ylim(0., np.pi)

    datamax = np.max(result)
    datamin = np.min(result)
    print(datamax,datamin)

    import matplotlib.colors as colors
    norm = colors.SymLogNorm(linthresh=1., vmin=datamin, vmax=datamax)
    #pcm = axes.pcolormesh(r_ax, theta_ax, result, norm=norm, cmap=cmap, vmin=datamin, vmax=datamax)
    pcm = axes.pcolormesh(r, theta, result, norm=norm, cmap=cmap, vmin=datamin, vmax=datamax)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    cb = axes.figure.colorbar(pcm, cax=cax)

    plt.savefig('laplace_colorplot.png')
    plt.clf()


def plot3d(r_min, r_max, r_crit, phi0, epsilonU, epsilonL, N=101, sig=9.):
    r_ax, theta_ax, result = solve(r_min, r_max, r_crit, phi0, epsilonU, epsilonL, N=N, sig=sig)

    r = np.kron(r_ax, np.ones(N)).reshape((N,N))
    theta = np.kron(np.ones(N), theta_ax).reshape((N,N))

    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    #print(sys.version)
    #print(sys.path)
    #print(os.environ['PYTHONPATH'].split(os.pathsep))

    fig = plt.figure()
    # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    ax = plt.axes(projection='3d') 

    ax.plot3D(r.flatten(), theta.flatten(), result.flatten(), 'b.', lw=2, alpha=0.7)

    ax.set(xlabel = "r")
    ax.set(ylabel = "theta")
    ax.set(zlabel = "phi (result)")

    ax.set_xlim(r_min-0.5, r_max+0.5)
    ax.set_ylim(0., np.pi)
    ax.set_zlim(-1., 4.)

    plt.show()

def export_vtk(r_min, r_max, r_crit, phi0, epsilonU, epsilonL, N=101, sig=9., fname=None, angle_axis=True):
    r_ax, theta_ax, phi = solve(r_min, r_max, r_crit, phi0, epsilonU, epsilonL, N=N, sig=sig)
    print('solved')

    r = np.kron(r_ax, np.ones(N))
    theta = np.kron(np.ones(N), theta_ax)
    phi = phi.flatten()

    if angle_axis:
        points = np.column_stack([r,theta,phi])
    else:
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        points = np.column_stack([x,y,phi])

    if fname is None:
        fname = base + 'solution_rmin_%f_rmax_%f_rcrit_%f_phi0_%f_epU_%f_epL_%f_aa_%s.vtk'%(r_min, r_max, r_crit, phi0, epsilonU, epsilonL, str(angle_axis))

    sys.path.append('/home/gary/magnetovis/pkg/magnetovis/')
    from vtk_export import vtk_export

    vtk_export(fname, points,
                    dataset = 'STRUCTURED_GRID',
                    connectivity = (N,N,1),
                    point_data = phi,
                    texture = 'SCALARS')

def export_pkl(r_min, r_max, r_crit, phi0, epsilonU, epsilonL, N=101, sig=9., fname=None):
    r_ax, theta_ax, phi = solve(r_min, r_max, r_crit, phi0, epsilonU, epsilonL, N=N, sig=sig)
    print('solved')

    dic = { 'r_ax' : r_ax,
            'theta_ax' : theta_ax,
            'phi' : phi,
            'N' : N}

    if fname is None:
        fname = base + 'solution_rmin_%f_rmax_%f_rcrit_%f_phi0_%f_epU_%f_epL_%f.pkl'%(r_min, r_max, r_crit, phi0, epsilonU, epsilonL)

    import pickle
    print('writing '+fname)
    with open(fname, 'wb') as handle:
        #https://stackoverflow.com/questions/29587179/load-pickle-filecomes-from-python3-in-python2/43290778
        pickle.dump(dic, handle, protocol=2)


def pkl_to_vtk(pkl, angle_axis=True):
    import pickle
    with open(pkl, 'rb') as handle:
        dic = pickle.load(handle)
    r_ax = dic['r_ax']
    theta_ax = dic['theta_ax']
    phi = dic['phi']
    N = dic['N']

    r = np.kron(r_ax, np.ones(N))
    theta = np.kron(np.ones(N), theta_ax)
    phi = phi.flatten()

    if angle_axis:
        points = np.column_stack([r,theta,phi])
    else:
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        points = np.column_stack([x,y,phi])

    sys.path.append('/home/gary/magnetovis/pkg/magnetovis/')
    from vtk_export import vtk_export

    vtk_export(pkl + '.vtk', points,
                    dataset = 'STRUCTURED_GRID',
                    connectivity = (N,N,1),
                    point_data = phi,
                    texture = 'SCALARS')


def testplot(r_ax, theta_ax, result, r_min, r_max, phi0):
    import matplotlib.pyplot as plt
    plt.plot(r_ax, result[:,20], '.')
    c2 = phi0/(1./r_min - 1./r_max)
    c1 = -c2/r_max
    print(c1,c2)
    plt.plot(r_ax, c1 + c2/r_ax)
    print('saving plot as CT_lap_testplot.png')
    plt.savefig('CT_lap_testplot.png')
    plt.show()
    plt.clf()


def test():
    N = 1000
    dtheta = np.pi/(N+1)
    theta = np.linspace(dtheta, np.pi-dtheta, N)
    assert(np.allclose( eta2theta(theta2eta(theta)) , theta ))

    sig = 9.
    eta = np.linspace(-sig,sig, N)
    assert(np.allclose( theta2eta(eta2theta(eta)) , eta ))

    solve_noDielectric(1., 20., 3., N=3, debug=True)
    print('\n ############################### \n')
    solve(1., 20., 1., 3., 2., 4., N=3, sig=9., debug=True)
    print('\n ############################### \n')

    N=101
    r_ax, theta_ax, phi = solve_noDielectric(1., 20., 3., N=N, sig=9., debug=False)

    for j1 in range(N):
        for j2 in range(N):
            #print(np.max(np.abs(phi[:,j1] - phi[:,j2])))
            assert(np.allclose(phi[:,j1], phi[:,j2]))

    testplot(r_ax, theta_ax, phi, 1., 20., 3.)


def main():
    #test()
    #colorplot(1., 20., 2., 3., 2., 4., N=101, sig=9.)
    #plot3d(1., 20., 2., 3., 2., 4., N=101, sig=9.)
    #export_pkl(1., 20., 2., 3., 2., 4., N=1001, sig=9.)
    export_vtk(1., 20., 2., 3., 2., 4., N=1001, sig=9.)
    #pkl_to_vtk(base + 'CT_lap.pkl')
    #print('main')
    #M = ssp.csr_matrix([[0,1.5,0],[0,0,1.5],[0,0,0]])
    #print(M.toarray())
    #print(type(M[0,:]))
    #print(M[0,:].indices)
    #M[0,0]=3.4
    #print(M.toarray())


if __name__ == '__main__':
    main()
