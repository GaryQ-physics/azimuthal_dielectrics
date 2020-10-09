import numpy as np


def forwarddir_matrix(N,  bc='neumann'):
    diff = np.zeros((N, N))
    for n in range(N):
        if n==0:

            if bc=='dirichlet':
                diff[0, 0] = -1.
                diff[0, 1] = 1.
            elif bc=='neumann':
                diff[0, 0] = -1.
                diff[0, 1] = 1.

        elif n==N-1:

            if bc=='dirichlet':
                diff[N-1, N-1] = -1.
                #diff v [N-1] =  -v[N-1] + 0 ("0==v[N]") = -0.5*v[N-2]
            elif bc=='neumann':
                pass
                #diff v [N-1] =  -v[N-1] + v[N-1] ("v[N-1]==v[N]") = 0

        else:
            diff[n, n] = -1.
            diff[n, n+1] = 1.
            #diff v [n] = -v[n] + v[n+1]

    return diff

def backwarddir_matrix(N,  bc='neumann'):
    diff = np.zeros((N, N))
    for n in range(N):
        if n==0:

            if bc=='dirichlet':
                diff[0, 0] = 1.
            elif bc=='neumann':
                pass
        elif n==N-1:

            if bc=='dirichlet':
                #diff[N-1, N-2] = -1.
                #diff[N-1, N-1] = 1.
                diff[N-1, N-1] = -1.

            elif bc=='neumann':
                diff[N-1, N-2] = -1.
                diff[N-1, N-1] = 1.

        else:
            diff[n, n-1] = -1.
            diff[n, n] = 1.
            #diff v [n] = -v[n] + v[n+1]

    return diff


def symdir_matrix(N,  bc='neumann'):
    diff = np.zeros((N, N))
    for n in range(N):
        if n==0:

            if bc=='dirichlet':
                diff[0, 1] = 0.5
                #diff v [0] =  -0.5*0 + 0.5*v[1] ("0==v[-1]") = 0.5*v[1]
            elif bc=='neumann':
                print('blaaa')
                diff[0, 0] = -1.#-0.5
                diff[0, 1] = 1.#0.5
                #diff v [0] =  -0.5*v[0] + 0.5*v[1] ("v[0]==v[-1]")

        elif n==N-1:

            if bc=='dirichlet':
                diff[N-1, N-2] = -0.5
                #diff v [N-1] =  -0.5*v[N-2] + 0.5*0 ("0==v[N]") = -0.5*v[N-2]
            elif bc=='neumann':
                diff[N-1, N-2] = -1.#-0.5
                diff[N-1, N-1] = 1.#0.5
                #diff v [N-1] =  -0.5*v[N-2] + 0.5*v[N-1] ("v[N-1]==v[N]")

        else:
            diff[n, n-1] = -0.5
            diff[n, n+1] = 0.5
            #diff v [n] = -0.5*v[n-1] + 0.5*v[n+1]

    return diff

def secondir_matrix(N, bc='neumann'):
    diff = np.zeros((N, N))
    for n in range(N):
        if n==0:

            if bc=='dirichlet':
                diff[0, 0] = -2.
                diff[0, 1] = 1.
                #diff v [0] =  +1*0 -2*v[0] + 1*v[1]("0==v[-1]") = -2*v[0] + 1*v[1]
            elif bc=='neumann':
                diff[0, 0] = -1.
                diff[0, 1] = 1.
                #diff v [0] = + 1*v[0] -2*v[0] + 1*v[1]  ("v[0]==v[-1]") = -2*v[0] + 1*v[1]

        elif n==N-1:

            if bc=='dirichlet':
                diff[N-1, N-2] = 1.
                diff[N-1, N-1] = -2.
                #diff v [N-1] =  +1*v[N-2] -2*v[N-1] + 1*0("0==v[N]")
            elif bc=='neumann':
                diff[N-1, N-2] = 1.
                diff[N-1, N-1] = -1.
                #diff v [N-1] = + 1*v[N-2] -2*v[N-1] + 1*v[N-1]  ("v[N-1]==v[N]") = 1*v[N-2] + -1*v[N-1]

        else:
            diff[n, n-1] = 1.
            diff[n, n] = -2.
            diff[n, n+1] = 1.
            #diff v [n] = + 1*v[n-1] -2*v[n] + 1*v[n+1]
    return diff


if __name__ == '__main__':
    N = 99

    f_R = 1.
    dx = 1./(N+1)
    A = forwarddir_matrix(N,  bc='dirichlet')/dx + np.identity(N)

    b = np.zeros(N)
    b[N-1] = -f_R/dx

    x = np.linspace(dx,1.-dx,N)
    #print(x)
    f = np.linalg.solve(A,b)
    f_an = np.exp(1-x)

    if False:
        import matplotlib.pyplot as plt
        plt.plot(x, f)
        plt.plot(x, f_an)
        plt.show()

    f_L = 1.5

    print((f[0] - f_L)/dx + f_L == 0. )
    print((f[0] - f_L)/dx + f_L)

if False:
    R_min = 10.
    R_max = 100.
    Nt = 40
    Nr = 80

    #print(dir_matrix(Nt))

    #print(secondir_matrix(Nt))

    #print(np.matmul(dir_matrix(Nt), dir_matrix(Nt)))


    theta_ax = np.linspace(0, np.pi, Nt+2)[1:-1]
    r_ax = np.linspace(R_min, R_max, Nr+2)[1:-1]

    dtheta = np.pi/(Nt+1)
    dr = (R_max-R_min)/(Nr+1)

    r = np.diag(r_ax)
    cottheta = np.diag(1./np.tan(theta_ax))

    d_dr = backwarddir_matrix(Nr, bc='dirichlet')/dr
    d2_dr2 = secondir_matrix(Nr, bc='dirichlet')/(dr**2)
    d_dtheta = backwarddir_matrix(Nt)/dtheta
    d2_dtheta2 = secondir_matrix(Nt)/(dtheta**2)

    r_op = np.matmul(r**2, d2_dr2) + np.matmul(2*r, d_dr)
    theta_op = d2_dtheta2 + np.matmul(cottheta, d_dtheta)

    #print(r_op)
    #print(theta_op)

    print('hello')
    rtominus2 = np.diag(r_ax**-2)
    laplacian = np.kron(rtominus2, theta_op) + np.kron(np.matmul(rtominus2,r_op), np.identity(Nt))
    # scaledlap = r**2 laplacian
    scaledlap = np.kron(np.identity(Nr), theta_op) + np.kron(r_op, np.identity(Nt))
    print('there')

    #print(scaledlap)
    #print(scaledlap.shape)

    phi0 = 3.
    r_innerbc = phi0*np.ones(Nt)

    #inhomo = -R_min**2*np.kron(np.ones(Nr), r_innerbc)
    inhomo = -np.kron(np.ones(Nr), r_innerbc)

    inv = np.linalg.inv(laplacian)
    print(np.min(np.abs(inv)))
    print(np.max(np.abs(inv)))
    result = np.matmul(inv, inhomo)

    #print(result)

    result = result.reshape((Nr,Nt))

    print(result[:,10])

    import matplotlib.pyplot as plt
    plt.plot(r_ax, result[:,10], '.')

    plt.show()


