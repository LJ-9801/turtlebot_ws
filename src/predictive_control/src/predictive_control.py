#!/usr/bin/env python3
from cmath import pi
import math as mt
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid

def callback(msg):
    map = msg.data
    rospy.loginfo(map[0])


def main():
    rospy.init_node('predictive_control')
    rospy.Subscriber("/map", OccupancyGrid, callback)
    #rate = rospy.Rate(hz)
    rospy.spin()
    #while not rospy.is_shutdown():
        
if __name__ == '__main__':
    main()


def handle(hI, eta, c, m, e):
    y = []
    for i in range(hI):
        x
        if(hI[i] > eta):
            x = hI[i]*(-m*mt.atan(c*(hI[i]-eta)*2/pi))
    return x
    

def f_func(x):
    f = np.matrix([[mt.cos(x[2]), 0],[mt.sin(x[2]), 0],[0, 1]])
    return f

def fu_func(x, u):
    f = f_func(x)
    fu = f @ u
    return fu

def grad_fu(x, u, ts):
    n = np.prod(x.shape)
    A = np.eye(n) +  ts * np.matrix([[0, 0, -float(u[0])*mt.sin(x[2])],
                                  [0, 0, float(u[0])*mt.cos(x[2])],
                                  [0, 0, 0]])
    
    B = ts*f_func(x)

    return [A, B]

def fu_disc(x, u, ts):
    #n = np.prod(x.shape)
    x_new = x + ts * fu_func(x, u)
    return x_new.T

def traj(x0, ubar, m, ts):
    n = np.prod(x0.shape)
    Nm = np.prod(ubar.shape)
    k = int(Nm/m)
    x = np.zeros((n, k+1))
    u = np.reshape(ubar, (k, m)).T
    x[:,0] = x0.T

    for i in range(k):
        x_new = fu_disc(x[:, i], u[:, i], ts)
        x[:, i+1] = x_new.T

    
    tmp = x[:, 1:k+1].T
    xbar = np.reshape(tmp, (n*k, 1))
    return xbar


def trajgrad(xbar, ubar, n, m, ts):
    Nm = np.prod(ubar.shape)
    k = int(Nm/m)
    u = np.reshape(ubar, (k, m)).T
    x = np.reshape(xbar, (k, n)).T
    J = np.zeros((n, k*m))
    for i in range(k):
        [A,B] = grad_fu(x[:, i], u[:, i], ts)
        J = A @ J
        J[:, i*m:m*i+2]=B
    
    return J


def trajgrad_k(j, xbar, ubar, n, m, ts):
    Nm = np.prod(ubar.shape)
    k = int(Nm/m)
    u = np.reshape(ubar, (k, m)).T
    x = np.reshape(xbar, (k, n)).T
    J_k = np.zeros((n, k*m))
   
    for i in range(j):
        [A, B] = grad_fu(x[:, i], u[:, i], ts)
        J_k = A @ J_k
        J_k[:, i*m:m*i+2] = B

    return J_k
