#!/usr/bin/env python3
from cmath import pi, sqrt
import math as mt
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid,Odometry
import message_filters


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


def colcheck(res, x_o, y_o, grid, q_x, q_y):
    width = 384
    height = 384
    dist = 1000
    d = []
    for i in range(len(grid)):
        if(grid[i] > 90):
            x = i // width
            y = i % height
            dx = (q_x - (x * res + x_o))
            dy = (q_y - (y * res + y_o))
            dmin = mt.sqrt(dx**2 + dy**2)
            if(dmin < dist):
                dist = dmin
                d = [dx, dy]
    
    return dist, d
            
            




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
    
    
    #rospy.spin()
        
if __name__ == '__main__':
    main()


def handle(hI, eta, c, m, e):
    y = []
    for i in range(hI):
        x
        if(hI[i] > eta):
            x = hI[i]*(-m*mt.atan(c*(hI[i]-eta)*2/pi))
    return x
    

