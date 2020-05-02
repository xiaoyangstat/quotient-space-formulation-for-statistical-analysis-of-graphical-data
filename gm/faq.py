# -*- coding: utf-8 -*-
"""fast approximate quadratic programming

Rewrote from https://github.com/jovo/FastApproximateQAP;
Support additional node attributes for matching

Created on Mon Mar 27 2020

@author: J. Derek Tucker, Xiaoyang Guo
"""

import numpy as np
import numpy.matlib
from scipy.sparse import spdiags, eye
from scipy.optimize import linprog
import warnings
import lap


def assign(A, munk=True):
    """
    optional second if set to false will use the
    maxassing_linprog()
    """
    if munk == False:
        p,w,x = maxassign_linprog(A.transpose())
        p = p.transpose()
    else:
        ##debug##
        # import scipy.io as sio
        # sio.savemat('test.mat', {'d':A})
        ##
        #p,w,v,u,costMat = lapjv(-A.transpose(), 0.01) # from original implementation
        costMat, p, w = lap.lapjv(-A.transpose()) # from https://github.com/gatagat/lap
        w = -1*w
        x = perm2mat(p)
        x = x.transpose()

    return p,w,x

def dsproj(x,g,m,n):
    """
    minimize gradient times d subject to linear
    constraints
    """
    P,Q = unstack(x,m,n)
    gP,gQ = unstack(g,m,n)

    # import scipy.io
    # scipy.io.savemat('test.mat', mdict={'d': -gQ})

    q,wq,wQ = assign(-gQ)
    wP = wQ
    w = stack(wP,wQ,m,n)
    d = w-x
    return d,q

def stack(P,Q,m,n):
    """
    stack A,B -> x
    """
    a = P.reshape(m*m,1,order='F').copy()
    b = Q.reshape(m*m,1,order='F').copy()
    x = np.vstack((a,b))

    return x

def unstack(x,m,n):
    """
    unstack x -> P,Q
    """
    tmp = x[0:(m*m)]
    P = tmp.reshape(m,m,order='F').copy()
    tmp = x[(m*m):(m*m+n*n)]
    Q = tmp.reshape(n,n,order='F').copy()

    return P,Q

def stoch(A, **kwargs):
    """
    Normalize A so that is row stochastic if one
    input argument is given or along the dim
    dimension 0 for column stochastic and 1 for
    row stochastic.

    A = stoch(A,dim)

    Currently only 2D arrays are supported
    """
    m,n = A.shape
    varargin = kwargs
    nargin = 1 + len(varargin)
    realmin =  np.finfo(float).tiny
    if nargin > 1:
        dim = kwargs['dim']
    if ((nargin == 1) or (dim==1)):
        s = A.sum(axis=1)
        s = s.todense()
        if np.any(s) == 0:
            s = np.maximum(s,realmin)
            warnings.warn("Zero sum found!")

        A = np.matmul(spdiags(1.0/s,0,m,m),A)
    else:
        s = A.sum(axis=0)
        s = s.todense()
        s = s.transpose()
        if np.any(s) == 0:
            s = np.maximum(s,realmin)
            warnings.warn("Zero sum found!")
        A = np.matmul(A,spdiags(1.0/s,0,n,n))

    return A,s

def sink(A,n):
    """
    perform n interations of Sinkhorn balancing
    """
    for l in range(0,n):
        A = stoch(A,dim=0)
        A = stoch(A,dim=1)

    return A

def perm2mat(p):
    n = max(p.shape)
    P = np.zeros((n,n))
    for i in range(0,n):
        P[i,p[i]] = 1

    return P

def maxassign_linprog(C):
    """
    find the maximum assignment of C where p and
    w are such that w = 0 for i = 1:n, w=w+C(i,p(i)) end;
    and x is the interior point solution given by
    the linear program, which is reshaped to an
    n by n matrix should be close to a permutation
    matrix
    """
    m,n = C.shape
    if m > n:
        raise ValueError('Matrix cannot have more rows than columns')

    # put in zeros to square out matrix so as not
    # to affect the score or assignment
    C = np.vstack((C,np.zeros((n-m,n))))
    a1 = np.kron(eye(n),np.ones((1,n)))
    a2 = np.kron(np.ones((1,n)),eye(n))
    A = np.vstack((a1,a2))
    A = A[0:-1,:]
    b = np.ones((2*n-1,1))
    c = -C.flatten('F')
    res = linprog(c,A_eq=A,b_eq=b,bounds=(0,1))
    x = res.x
    X = x.reshape(n,n,order='F')
    x = X.transpose()
    p = x.argmax(axis=0)
    p = p[0:m]
    w = -res.fun

    return p,w,x

def fun(x,A,B,D):
    m,n = A.shape
    P,Q = unstack(x,m,n)
    f0 = np.sum(np.sum(np.multiply((P@A@Q.transpose()),B)))
    f0 = f0 + np.trace(D)
    
    return f0

def fungrad(x,A,B,D):
    m,n = A.shape
    P,Q = unstack(x,m,n)
    f0 = fun(x,A,B,D)
    tmp = B@Q@A.transpose()+D
    g = tmp.reshape(m*m,1,order='F').copy()
    tmp = B.transpose()@P@A+D
    g1 = tmp.reshape(n*n,1,order='F').copy()
    g = np.vstack((g,g1))
    return f0, g

def lines(ltype,x,d,g,A,B,D):
    """
    line search 0 => salpha=1, 1=>search
    """
    nT = max(g.shape)
    dxg = (d.transpose()@g)[0,0]

    if dxg > 0:
        message = 'Nonimproving Direction, <d,g> = {}'.format(dxg)
        warnings.warn(message)

    if ltype == 0:
        salpha = 1
    elif ltype == 1:
        LARGE = 7.7e77
        l_TOL1 = 2.0e-12
        l_iter = 0
        alpha = 0.0
        alpha_u = 0.0
        alpha_c = 0.0
        Done = 0
        nu = 1.25
        leps = 0.2

        # alpha_c (CONSTRAINED SEARCH: Find Boundary)
        alpha_c = LARGE
        tdelta = alpha_c

        # Simplex constraint search
        for j in range(0,nT):
            if (d[j,0] < -l_TOL1):
                tdelta = -x[j,0] / d[j,0]
                alpha_c = np.minimum(tdelta,alpha_c)

        if alpha_c == 0.0:
            print("alpha_c Error: alpha_c = %f\n" % alpha_c)
        elif alpha_c == LARGE:
            print("Constrained Line Search Error")

        F0_0 = fun(x,A,B)
        xt = x + alpha_c * d
        F0_c = fun(xt,A,B)

        # bisection search
        Done = 0
        alpha_u = alpha_c
        F0_R = F0_c
        alpha_n = 1.0
        l1 = alpha_c / 2.0
        xt = x + l1 * d
        F0_L = fun(xt,A,B)
        half = 2.00

        while (not Done):
            l_iter += 1
            if l_iter > 25:
                print("Error: Too many line searches = %d, alpha %f\n" % (l_iter,l1))

            if F0_L < F0_R:
                F0_R = F0_L
                alpha_u = l1
                l1 /= half
                xt = x + l1 * d
                F0_L = fun(xt,A,B)
            else:
                if F0_R < F0_0:
                    F0_u = F0_R
                    Done = 1
                else:
                    l1 /= half
                    xt = x + l1 * d
                    F0_L = fun(xt,A,B)

        # cleanup
        # compute alpha
        if F0_c <= F0_u:
            F0 = F0_c
            salpha = alpha_c
            alpha_n = 1.0
            exact = 0
        else:
            F0 = F0_u
            salpha = alpha_u
            alpha_n = alpha_u/alpha_c
            exact = 1

        mu = alpha_c
        if alpha < 0.0:
            print("error: alpha < 0, alpha = %f\n" % alpha)

        tmp = np.abs((F0-F0_0)/F0)
        if ((tmp > 1.0e-12) and (F0 > F0_0)):
            print("Nonmonotone Line Search Error: cost > last cost, cost = %f, last cost = %f" %(F0,F0_0))

    elif ltype == 2:
        # assume the function is quadratic
        # derivative at alpha = 0
        b = (g.transpose()@d)[0,0]
        # constant term at alpha = 0
        c = fun(x,A,B,D)
        # get second order coeff
        fun_vertex = fun(x+d,A,B,D)
        a = fun_vertex - b - c
        eps = np.finfo(np.double).eps
        if np.abs(a) < eps:
            salpha = 1
        else:
            salpha = np.minimum(1,np.maximum(-b/(2*a),0))

        fun_alpha = fun(x+salpha*d,A,B,D)
        # check quadratic function
        qfun_alpha = a*salpha*salpha + b*salpha + c
        if ((np.abs(a)>=eps) and (np.abs(fun_alpha-qfun_alpha)>1000*abs(fun_alpha)*eps)):
            pass
            #print('quadratic search error %d,%g\n' % (fun_alpha,qfun_alpha))

        if fun_alpha>c:
            salpha = 0
            fun_alpha = c

        if fun_alpha > fun_vertex:
            salpha = 1
            fun_alpha = fun_vertex

        f0new = fun_alpha
    else:
        print('unsuported line\n')

    xt = x + salpha * d
    f0new = fun(xt,A,B,D)

    return(f0new,salpha)

def lapjv(costMat, resolution=None):
    """Jonker-Volgenant Algorithm for Linear Assignment Problem

    ROWSOL, assigned to row in solution, and the minimum COST based on the
    assignment problem represented by the COSTMAT, where the (i,j)th element
    represents the cost to assign the jth job to the ith worker.
    Other output arguments are:
    v: dual variables, column reduction numbers.
    u: dual variables, row reduction numbers.
    rMat: the reduced cost matrix.

    For a rectangular (nonsquare) costMat, rowsol is the index vector of the
    larger dimension assigned to the smaller dimension.

    Args:
        resolution: the minimum resolution to differentiate costs between
        assignments. The default is eps.

    """
    if resolution==None:
        resolution = np.spacing(costMat.max())

    # prepare working data
    rdim,cdim = costMat.shape
    M = costMat.min()
    if rdim > cdim:
        costMat = costMat.transpose()
        rdim,cdim = costMat.shape
        swapf=True
    else:
        swapf=False

    dim = cdim
    costMat = np.vstack((costMat,2*M+np.zeros((cdim-rdim,cdim))))
    tmp = costMat[costMat < np.Inf]
    maxcost = tmp.max()*dim+1
    if maxcost.size == 0:
        maxcost = np.inf

    costMat[costMat==np.Inf] = maxcost
    v = np.zeros((1,dim))
    rowsol = np.zeros((1,dim), dtype=np.int)-1
    colsol = np.zeros((1,dim), dtype=np.int)-1
    
    if costMat.std(ddof=1) < costMat.mean():
        numfree = -1
        free = np.zeros((1,dim), dtype=np.int)-1
        matches = np.zeros((1,dim), dtype=np.int)

        # reverse order gives better results
        for j in range(dim-1,-1,-1):
            v[0,j] = costMat[:,j].min()
            imin = costMat[:,j].argmin()
            if not matches[0,imin]:
                rowsol[0,imin] = j
                colsol[0,j] = imin
            elif v[0,j] < v[0,rowsol[0,imin]]:
                j1 = rowsol[0,imin]
                rowsol[0,imin] = j
                colsol[0,j] = imin
                colsol[0,j1] = -1
            else:
                colsol[0,j] = -1

            matches[0,imin] += 1

        # reduction transfer from unassigned to assigned rows
        for i in range(0,dim):
            if not matches[0,i]:
                numfree += 1
                free[0,numfree] = i
            else:
                if matches[0,i] == 1:
                    j1 = rowsol[0,i]
                    x = costMat[i,:] - v
                    x = x.reshape((1,-1))
                    x[0,j1] = maxcost
                    v[0,j1] = v[0,j1] - x.min()
    else:
        numfree = dim - 1
        v1 = costMat.min(axis=0)
        r = costMat.argmin(axis=0)
        free = np.arange(0,dim)
        c = v1.argmin()
        r = r.reshape((1,-1))
        imin = r[0,c]
        j = c
        rowsol[0,imin] = j
        colsol[0,j] = imin
        free = np.delete(free,imin)
        free = free.reshape((1,-1))
        x = costMat[imin,:] - v
        x = x.reshape((1,-1))
        x[0,j] = maxcost
        v[0,j] = v[0,j] - x.min()

    # augmenting reduction of unassigned rows
    loopcnt = 0
    while loopcnt < 2:
        loopcnt += 1
        # scan all free rows
        # in some cases, a free row may be replaced
        # with another one to be scaned next
        k = 0
        prvnumfree = numfree
        numfree = 0
        while k < prvnumfree:
            k += 1
            i = free[0,k-1]
            # find minimum and second minimum reduced cost over columns
            x = costMat[i,:] - v
            umin = x.min()
            j1 = x.argmin()
            x = x.reshape((1,-1))
            x[0,j1] = maxcost
            usubmin = x.min()
            j2 = x.argmin()
            i0 = colsol[0,j1]
            if (usubmin - umin) > resolution:
                # change the reduction of the min column to the increase the min
                # reduced cost in the row to the subminimum
                v[0,j1] = v[0,j1] - (usubmin - umin)
            else:
                if i0 >= 0:
                    j1 = j2
                    i0 = colsol[0,j2]

            # reassign i to j1, possibly de-assigning an i0
            rowsol[0,i] = j1
            colsol[0,j1] = i
            if i0 >= 0 :
                if (usubmin - umin) > resolution:
                    # put in current k, and go back to that k
                    free[0,k-1] = i0
                    k -= 1
                else:
                    # print(k,numfree)
                    numfree += 1
                    free[0,numfree-1] = i0
    # augmentation phase
    for f in range(0,numfree):
        # print('f',f,free)
        freerow = free[0,f]
        d = costMat[freerow,:] - v
        pred = freerow*np.ones((1,dim))
        collist = np.arange(0,dim).reshape((1,-1))
        # print('collist',collist)
        low = 0
        up = 0
        unassignedfound = False
        while not unassignedfound:
            if up == low:
                last = low - 1
                d = d.reshape((1,-1))
                minh = d[0,collist[0,up]]
                up += 1
                for k in range(up,dim):
                    j = collist[0,k]
                    h = d[0,j]
                    if h <= minh:
                        if h < minh:
                            up = low
                            minh = h

                        collist[0,k] = collist[0,up]
                        collist[0,up] = j
                        up += 1

                # check if any of the min col are unassigned
                for k in range(low,up):
                    if colsol[0,collist[0,k]] < 0:
                        endofpath = collist[0,k]
                        unassignedfound = True
                        break
            if not unassignedfound:
                # update distances between freerow and all unscanned columns
                # via next scanned column
                j1 = collist[0,low]
                low += 1
                i = colsol[0,j1]
                x = costMat[i,:] - v
                h = x[0,j1] - minh
                xh = x - h
                k = np.arange(up,dim)
                j = collist[0,k]
                vf0 = xh < d
                vf = vf0[0,j]
                j = j.reshape((1,-1))
                # print('j',j)
                vj = j[0,np.ravel(vf)]
                vj = vj.reshape((1,-1))
                k = k.reshape((1,-1))
                vk = k[0,np.ravel(vf)]
                pred[0,vj] = i
                v2 = xh[0,vj]
                d[0,vj] = v2
                vf = v2 == minh
                vj = vj.reshape((1,-1))
                j2 = vj[0,np.ravel(vf)]
                vk = vk.reshape((1,-1))
                k2 = vk[0,np.ravel(vf)]
                cf = colsol[0,j2] < 0
                if np.any(cf):
                    i2 = np.where(cf)[0][0]
                    endofpath = j2[i2]
                    unassignedfound = True
                else:
                    i2 = cf.size + 1
                for k in range(0,i2-1):
                    collist[0,k2[k]] = collist[0,up]
                    collist[0,up] = j2[k]
                    up += 1

        # update column prices
        j1 = collist[0:(last+1)]
        v[0,j1] = v[0,j1] + d[0,j1] - minh
        # reset row and column assignments along the alternating path
        # print(pred,freerow)
        while True:
            i = pred[0,endofpath]
            colsol[0,endofpath] = i
            j1 = endofpath
            endofpath = rowsol[0,int(i)]
            rowsol[0,int(i)]=j1
            if (i == freerow):
                break
            
        # print(collist)

    rowsol = rowsol[0,0:rdim]
    tmpv = v[0,rowsol]
    u = np.diag(costMat[:,rowsol]) - tmpv.transpose()
    u = u[0:rdim]
    v = v[0:cdim]
    cost = u.sum()+np.sum(v[0,rowsol])
    costMat = costMat[0:rdim,0:cdim]
    utmp = np.matlib.repmat(u,cdim,1).transpose()
    vtmp = np.matlib.repmat(v,rdim,1)
    costMat = costMat - utmp - vtmp
    if swapf:
        costMat = costMat.transpose()
        t = u.transpose()
        u = v.transpose()
        v = t

    if cost > maxcost:
        cost = np.inf

    return rowsol,cost,v,u,costMat

def sfw(A,B,D=None,IMAX=30,x0=None):
    """
    perform at most IMAX iterations of the Frank-Wolfe
    method to compute an approximate solution to the
    quadratic assignment problem given the matrices
    A and B. A and B should be square and the same size.
    The method sees a permutation p which minimizes
        f(p)=sum(sum(A.*B(p,p))).
    Convergence is declared if a fix point is
    encountered or if the projected gradient has a
    2-norm of 1.0e-4 or less.

    IMAX is optional with a default value of 30 iterations.
        If IMAX is set to 0.5 then one iteration of FW is performed with no line search. Thi is Care Priee's LAP approximation to the QAP

    The starting point is optional as well and its default value is ones(n)/n, the flat double stochastic matrix.
    x0 may also be -1 which signifes a "random" starting point should be used.
        Here the start is given by 0.5*ones(n)/n + sing(rand(n),10) where sink(rand(n),10) performs 10 iterations of Sinkhorn balancing on a matrix whose entries are drwan from the uniform distribution on [0,1].
    x0 may also be a user specified by n by n double stochastic matrix.
    x0 may be a permuation vector of size n.

    On output:
        f=sum(sum(A.*B(p,p))), where
        p is the permutation found by FW after projecting the interior point
        to the boundary.
        x is the doubly stochastic matrix (interior point) computed by the FW
        method
        iter is the number of iterations of FW performed.
        fs is the list of fs for each iteration
        myps is the list of myps for each iteration

    """
    m,n = A.shape
    stype = 2
    if x0 == None:
        if IMAX == 0.5:
            t = np.eye(m)
            x = np.vstack((t.ravel('F'),t.ravel('F')))
        else:
            x0 = np.vstack((1/m*np.ones((m**2,1)),1/n*np.ones((n**2,1))))
    elif x0 == -1:
        X = np.ones(m)/m
        lam = 0.5
        Y = X.copy()
        x0 = np.vstack((X.ravel('F'),Y.ravel('F')))
    elif x0.size == A.shape[0]:
        x0 = perm2mat(x0)
        x0 = x0.transpose()
        x0 = np.vstack((x0.ravel('F'),x0.ravel('F')))
    elif x0.size == A.size:
        x0 = np.vstack((x0.ravel('F'),x0.ravel('F')))
    else:
        x0 = x0.ravel('F')

    x = x0.copy()

    stoptol = 1.0e-4
    myp = np.array([])
    iter = 0
    stop = 0
    myps = np.empty((np.int(np.ceil(IMAX)),n))
    myps.fill(np.nan)
    while ((iter < IMAX) and (stop == 0)):
        # fun + grad
        f0,g = fungrad(x,A,B,D)
        g1 = g[0:n**2] + g[(n**2):]
        g2 = g[0:n**2] + g[(n**2):]
        g = np.vstack((g1,g2))/2

        # projection
        d,myp = dsproj(x,g,m,n)
        stopnorm = ((d.transpose()@d)**0.5)[0,0]

        # stop rule
        if (stopnorm < stoptol):
            stop = 1

        # line search
        if IMAX > 0.5:
            f0new, salpha = lines(stype,x,d,g,A,B,D)
        else:
            salpha = 1

        x = x + salpha*d
        iter += 1
        if salpha == 0:
            stop = 1

        # if nargout > 4  @todo
        # P,Q = unstack(x,m,n)
        # if salpha != 1:
        #     temp = assign(P,1)
        # else:
        #     temp = myp

        # myps[iter,:] = temp
        # tmp = A*B[temp,temp]
        # fs[iter] = tmp.sum()
        # fn[iter] = np.linalg.norm(-1*A-B[temp,temp], 'fro')

    if salpha != 1:
        P,Q = unstack(x,m,n)
        myp,_,_ = assign(P,1)

    tmp = A*B[myp][:,myp]
    f = tmp.sum()

    return myp
