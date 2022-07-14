#!/usr/bin/env python3
from cmath import pi, sqrt
import math as mt
from ossaudiodev import control_labels
from scipy import sparse
import qpsolvers
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid,Odometry
from geometry_msgs.msg import Twist
import message_filters
from qpsolvers import solve_qp



def quadprog(P,q,G,h,lb,ub,m,N):
    #print(qpsolvers.available_solvers)
    sP = sparse.csc_matrix(P)
    sG = sparse.csc_matrix(G)
    x = solve_qp(P=sP,q=q,G=sG,h=h,A=None,b=None,lb = lb,ub = ub,solver= 'osqp' ,initvals = np.zeros((m*N,1)))
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
        #map info
        self.map = grid.data
        self.x_o = grid.info.origin.position.x
        self.y_o = grid.info.origin.position.y
        self.res = grid.info.resolution

    def colcheck(self, q):
        width = 384
        height = 384
        dist = 1000
        d = np.zeros((2,1))
        for i in range(len(self.map)):
            if(self.map[i] > 90):
                x = i // width
                y = i % height
                dx = (q[0] - (x * self.res + self.x_o))
                dy = (q[1] - (y * self.res + self.y_o))
                dmin = mt.sqrt(dx**2 + dy**2)
                if(dmin < dist):
                    dist = dmin
                    d[0] = dx
                    d[1] = dy
        
        return dist, d

    def gradU1_rep(self, q, rho0, eta):
        e = 0.01
        dist, rho = self.colcheck(q)
        dist1, rho1 = self.colcheck(q+np.array([[1],[0],[0]])*e)
        dist2, rho2 = self.colcheck(q+np.array([[0],[1],[0]])*e)
        dist3, rho3 = self.colcheck(q+np.array([[0],[0],[1]])*e)

        gradrho = np.array([(dist1-dist)/e, (dist2-dist)/e, (dist3-dist)/e])
        
        if(dist < rho0):
            gradx = eta*(dist-rho0) * gradrho
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
            grad_d_x[:,i:i+1] = (rho-d)/epsilon
        
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


class Control_Loop(object):
    def __init__(self, grid, odom):
        self.u = np.zeros((2,1))
        self.grid = grid
        self.odom = odom

    def callback(self):
        rospy.loginfo("u is: %f, %f stop", self.u[0], self.u[1])

    def loop(self):
        q_x = self.odom.pose.pose.position.x
        q_y = self.odom.pose.pose.position.y

        quat_w = self.odom.pose.pose.orientation.w
        quat_x = self.odom.pose.pose.orientation.x
        quat_y = self.odom.pose.pose.orientation.y
        quat_z = self.odom.pose.pose.orientation.z

        roll,pitch,yaw = euler_from_quaternion(quat_x,quat_y,quat_z,quat_w)

        x_f = np.array([[-2],[-2],[2]])

        eta=.2
        c=100
        M=50
        repel=10

        #parameters
        dmin= 0.2
        N = 20
        #Nmax = 100
        ts = 0.1
        m = 2 #2 inputs
        n = 3 #3 outputs

        k_pot = 0.06

        #constraints  
        umax = np.array([[4],[2]])
        ubarmx = np.reshape((np.ones((1,20))*umax).T,(40,1))
        x3min = -pi
        x3max = pi

        u0 = 0.1*np.random.rand(m,N)
        u0bar = np.reshape(u0.T, (N*m,1))
        x0 = np.array([[q_x],[q_y],[yaw]])

        x = np.zeros((n,1))
        u = np.zeros((m,1))
        x = x0
        xbar = np.zeros((n*N,1))
        ubar = np.zeros((m*N,1))
        ubar = u0bar
        xbar = traj(x0, u0bar, m, ts)

        epsilon = 0.8
        Kp = np.diag([1,1,0.1])*0.5

        #for i in range(Nmax):
        J = trajgrad(xbar, ubar, n, m, ts)
        x_NstepAhead = xbar[len(xbar)-3:len(xbar)]

        '''
        formulate constriant right here
        '''
        #inequality constraint
        Aineq = np.zeros((N-1,m*N))
        bineq = np.zeros((N-1,1))
        #max
        Ax3ineqmx = np.zeros((N-1,m*N))
        bx3ineqmx = np.zeros((N-1,1))
        #min
        Ax3ineqmn = np.zeros((N-1,m*N))
        bx3ineqmn = np.zeros((N-1,1))
        # start predicting
        for j in range(1,N):
            x_subk = xbar[len(xbar)-3:len(xbar)]
            collisionCheck = Collision_Routine(self.grid, self.odom)
            dist, d = collisionCheck.colcheck(x_subk)
            
            grad_dx = collisionCheck.grad_numerical(d,x_subk)
            J_subk = trajgrad_k(j, xbar, ubar, n, m, ts)
            if(dist < dmin):
                Aineq[j-1,:] = -d.T @ grad_dx @ J_subk
                bineq[j-1] = colsafe(0.5*d.T@d, eta, c, M, repel)
                
            if(x_subk[2] < x3min + 0.05):
                Ax3ineqmn = -J_subk[2,:]
                bx3ineqmn = -colsafe(x_subk[2] - x3max, eta, c, M, repel)
            if(x_subk[2] > x3max - 0.05):
                Ax3ineqmn = J_subk[2,:]
                bx3ineqmn = -colsafe(x3max - x_subk[2], eta, c, M, repel)
        
        #form constraints
        A = np.concatenate((Aineq, Ax3ineqmn, Ax3ineqmx))
        B = np.concatenate((bineq, bx3ineqmn, bx3ineqmx))
        H = J.T @ J + epsilon * np.eye(m*N)
        F = J.T @ (Kp @ (x_NstepAhead-x_f))
        # waiting to download solver
        delta_ubar = quadprog(H, F, A, B, -ubarmx-ubar, ubarmx-ubar, m, N)
        delta_ubar = np.reshape(delta_ubar, (len(delta_ubar),1))
        ubar_next = ubar + delta_ubar
        #print(delta_ubar)
        pred_collision = 1
        alpha = 1
        pot_gain = 1
        E = epsilon * np.diagflat(np.reshape((np.ones((m,1))*np.arange(1,1+N*0.1,0.1)).T,(m*N,1)))
        ubar_pot = np.zeros((m*N,1))
        


        while(pred_collision > 0):
            pred_collision = 0
            u[0] = ubar_next[0]
            u[1] = ubar_next[1]
            self.u = u
            x = fu_disc(x, u, ts)
            x = np.reshape(x,(3,1))
            ubar = np.concatenate((ubar_next[m:],np.zeros((m,1))))
            xbar = traj(x, ubar, m, ts)
            

            for i in range(1,N-1):
                x_subk = xbar[i*n+3:i*n+6]
                dist, d = collisionCheck.colcheck(x_subk)
                #dist = 0
                if(dist < dmin / 1.2):
                    pred_collision = 1
                    J_k = trajgrad_k(i, xbar, ubar, n, m, ts)
                    ubar_pot = ubar_pot + (collisionCheck.gradU1_rep(x_subk, 2*dmin, k_pot) @ J_k).T
                
            if(pred_collision > 0):
                alpha = alpha/1.5
                H = alpha*J.T @ J + E
                H = 0.5*(H @ H.T)
                F = alpha * J.T @ (Kp @ (x_NstepAhead-x_f))
                delta_ubar = quadprog(H, F, A, B, -ubarmx-ubar, ubarmx-ubar, m, N)
                if(alpha < 0.1):
                    pot_gain = pot_gain*1.1
                    delta_ubar = delta_ubar + pot_gain*ubar_pot
                ubar_next = ubar+delta_ubar
                """
                finish one loop    
                """
        return self.u

def pub_vel(vel):
    pub_msg = Twist()
    pub_msg.linear.x = vel[0]
    pub_msg.angular.z = vel[0]

    return pub_msg


def main_loop(grid, odom):
    control_loop = Control_Loop(grid, odom)
    u = control_loop.loop()
    control_loop.callback()

    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

    vel = pub_vel(u)
    pub.publish(vel)



def main():
    rospy.init_node('predictive_control')

    map_sub = message_filters.Subscriber('/map', OccupancyGrid)
    odom_sub = message_filters.Subscriber('/odom', Odometry)
    ts = message_filters.ApproximateTimeSynchronizer([map_sub, odom_sub], 10000, 0.1, allow_headerless=True)

    ts.registerCallback(main_loop)
    rospy.spin()
    
        
if __name__ == '__main__':
    main()

    

