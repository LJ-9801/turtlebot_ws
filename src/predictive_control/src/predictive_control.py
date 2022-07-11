#!/usr/bin/env python3
from cmath import pi, sqrt
import imp
import math as mt
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid,Odometry
import message_filters
from qpsolvers import solve_qp


def quadprog(H,F,A,B,lb,ub,m,N):
    x = solve_qp(H,F,A,B,[],[],lb,ub,np.zeros((m*N,1)))
    return x


def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = mt.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = mt.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = mt.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z


def colsafe(hI, eta, c, m, e):
    y = []
    for i in range(len(hI)):
        c1 = int(hI[i]>eta)
        c2 = int(hI[i]>0)
        c3 = int(hI[i]<eta)
        c4 = int(hI[i]<=0)
        x = c1*(-m*mt.atan(c*(hI[i]-eta))*2/pi)+c2*c3*(e*(eta-hI[i])/eta)+c4*e
        y.append(x)

    return y


class Collision_Routine:
    def __init__(self, grid, odom):
        
        #robot state
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        quat_w = odom.pose.pose.orientation.w
        quat_x = odom.pose.pose.orientation.x
        quat_y = odom.pose.pose.orientation.y
        quat_z = odom.pose.pose.orientation.z
        roll,pitch,yaw = euler_from_quaternion(quat_x,quat_y,quat_z,quat_w)
        q = np.array([[x],[y],[yaw]])
        self.q = q

        #map info
        self.map = grid.data
        self.x_o = grid.info.origin.position.x
        self.y_o = grid.info.origin.position.y
        self.res = grid.info.resolution

    def colcheck(self, q):
        width = 384
        height = 384
        dist = 1000
        d = []
        for i in range(len(self.map)):
            if(self.map[i] > 90):
                x = i // width
                y = i % height
                dx = (q[0] - (x * self.res + self.x_o))
                dy = (q[1] - (y * self.res + self.y_o))
                dmin = mt.sqrt(dx**2 + dy**2)
                if(dmin < dist):
                    dist = dmin
                    d = [dx, dy]
        
        return dist, d

    def gradU1_rep(self, q, rho0, eta):
        e = 0.01
        dist, rho = self.colcheck(q)
        dist, rho1 = self.colcheck(q+np.array([[1],[0],[0]])*e)
        dist1, rho2 = self.colcheck(q+np.array([[0],[1],[0]])*e)
        dist2, rho3 = self.colcheck(q+np.array([[0],[0],[1]])*e)

        gradrho = np.array([(rho1-rho)/e, (rho2-rho)/e, (rho3-rho)/e])
        
        if(rho < rho0):
            gradx = eta*(rho-rho0)*gradrho
        else:
            gradx = np.array([0,0,0])

        return gradx

    def grad_numerical(self, d, x):
        nd = len(d)
        n = len(x)
        epsilon = 0.01
        grad_d_x = np.zeros((nd,n))
        for i in range(n):
            e_i = np.zeros((n,1))
            e_i[i] = 1
            d_epsilon, rho = self.colcheck(x+epsilon*e_i)
            grad_d_x[:,i:i+1] = (d_epsilon-d)/epsilon
        
        return grad_d_x
    


        


        

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

def callback(grid, odom):
    map = grid.data
    x = odom.pose.pose.position.x
    y = odom.pose.pose.position.y

    #grid origin
    x_o = grid.info.origin.position.x
    y_o = grid.info.origin.position.y

    #grid resolution
    res = grid.info.resolution

    quat_w = odom.pose.pose.orientation.w
    quat_x = odom.pose.pose.orientation.x
    quat_y = odom.pose.pose.orientation.y
    quat_z = odom.pose.pose.orientation.z

    roll,pitch,yaw = euler_from_quaternion(quat_x,quat_y,quat_z,quat_w)

    dist, d = colcheck(res, x_o, y_o, map, x, y)
    

    rospy.loginfo("x is: %f, y is: %f, yaw is : %f, distmin is: %f", x, y, yaw, dist)

def main():
    rospy.init_node('predictive_control')
    map_sub = message_filters.Subscriber('/map', OccupancyGrid)
    odom_sub = message_filters.Subscriber('/odom', Odometry)
    ts = message_filters.ApproximateTimeSynchronizer([map_sub, odom_sub], 1000, 0.1, allow_headerless=True)
    
    #rate = rospy.Rate(100)
    
    ts.registerCallback(callback)
    rospy.spin()
    
        
if __name__ == '__main__':
    main()


def handle(hI, eta, c, m, e):
    y = []
    for i in range(hI):
        x
        if(hI[i] > eta):
            x = hI[i]*(-m*mt.atan(c*(hI[i]-eta)*2/pi))
    return x
    

