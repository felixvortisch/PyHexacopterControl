# (c) Jan Zwiener und Felix Vortisch

import numpy as np
from acados_template import *
from casadi import SX, vertcat, sin, cos, cross

def export_rocket_ode_model() -> AcadosModel:

    model_name = 'copterpos_ode'

    # constants
    # ---------
    gravity = 9.81
    mass_kg = 4.606 
    
    # Anzahl der Zeilen und Spalten der Matrix
    num_rows_J = 3
    num_cols_J = 3

    # Moments of inertia
    J = SX.sym('J', num_rows_J, num_rows_J)
    J_inv = SX.sym('J_inv', num_rows_J, num_rows_J)

    # Werte für die symbolische Matrix festlegen
    values_J = [	[0.2566543715, 0.0, -0.00240975],
    			[0.0, 0.2502375625, 0.0],
    			[-0.00240975, 0.0, 0.363093409]]
    values_J_inv = [	[3.8965333, 0.0, 0.02586021],
                       [0.0, 3.9962026, 0.0],
                       [0.02586021, 0.0, 2.754283861]]
                  
    # Symbolische Variablen mit Werten füllen
    for i in range(num_rows_J):
        for j in range(num_cols_J):
            J[i, j] = values_J[i][j]
            J_inv[i, j] = values_J_inv[i][j]

    
    # Anzahl der Zeilen und Spalten der Matrix
    num_rows_M = 3
    num_cols_M = 6

    # Moments of inertia
    M = SX.sym('M', num_rows_M, num_cols_M)

    # torque from Motormatrix K
    values_M = [[5.6607, 11.3214, 5.6607, -5.6607, -11.3214, -5.6607],
		       [-9.8116, 0.0, 9.8116, 9.8116, 0.0, -9.8116],
		       [0.01, -0.01, 0.01, -0.01, 0.01, -0.01]]		#With rotated thust vectors 60° around Z Axis and 8° Azimut (DJI S900 specific)
    #values_M = [[5.5181, 11.0363, 5.5181, -5.5181, -11.0363, -5.5181],
	#	       [-9.5648, 0.0, 9.5648, 9.5648, 0.0, -9.5648],
	#	       [0.01, -0.01, 0.01, -0.01, 0.01, -0.01]]	       	#No rotation (Thrust vectors parallel to Z Axis)
		      


    # Symbolische Variablen mit Werten füllen
    for i in range(num_rows_M):
        for j in range(num_cols_M):
            M[i, j] = values_M[i][j]


    #Thrust vectors for each motor
    thrust_vectors = np.array([[-2.9559, 0.0, 2.9559, 2.9559, 0.0, -2.9559],
    				[-1.7066, -3.4132, -1.7066, -1.7066, 3.4132, -1.7066],
    				[24.2863,24.2863, 24.2863,24.2863,24.2863,24.2863]])	#rotated thust vectors 30° around Z Axis and 8° Azimut with force of 24.525Nm
    #thrust_vectors = np.array([[0.0,0.0,0.0,0.0,0.0,0.0],
    #				[0.0,0.0,0.0,0.0,0.0,0.0],
    #				[24.525,24.525, 24.525,24.525,24.525,24.525]])		#No rotation (Thrust vectors parallel to Z Axis)			

    U_MAX = 1.0
    U_MIN = 0.01
    U_TAU = 0.1


    # set up states & controls
    # x
    q_0         = SX.sym('q_0') # qw (quaternion from body to navigation frame)
    q_1         = SX.sym('q_1') # qx
    q_2         = SX.sym('q_2') # qy
    q_3         = SX.sym('q_3') # qz
    q           = vertcat(q_0, q_1, q_2, q_3)

    omega_x     = SX.sym('omega_x') # rotation rates body vs. navigation frame in body frame
    omega_y     = SX.sym('omega_y')
    omega_z     = SX.sym('omega_z')
    omega       = vertcat(omega_x, omega_y, omega_z)

    # xdot
    q0_dot      = SX.sym('q0_dot')
    q1_dot      = SX.sym('q1_dot')
    q2_dot      = SX.sym('q2_dot')
    q3_dot      = SX.sym('q3_dot')
    q_dot       = vertcat(q0_dot, q1_dot, q2_dot, q3_dot)

    omega_dot   = SX.sym('omega_dot', 3, 1)

    pos     = SX.sym('pos', 3, 1) # position in meter
    pos_dot = SX.sym('pos_dot', 3, 1)
    vel     = SX.sym('vel', 3, 1) # velocity in m/s
    vel_dot = SX.sym('vel_dot', 3, 1)

    
    u_1     = SX.sym('u_1')
    u_2     = SX.sym('u_2')
    u_3      = SX.sym('u_3')
    u_4      = SX.sym('u_4')
    u_5      = SX.sym('u_5')
    u_6      = SX.sym('u_6')
    u_old       = vertcat(u_1, u_2, u_3, u_4, u_5, u_6)
    
    u1_dot      = SX.sym('u1_dot')
    u2_dot      = SX.sym('u2_dot')
    u3_dot      = SX.sym('u3_dot')
    u4_dot      = SX.sym('u4_dot')
    u5_dot      = SX.sym('u5_dot')
    u6_dot      = SX.sym('u6_dot')
    u_old_dot       = vertcat(u1_dot, u2_dot, u3_dot, u4_dot, u5_dot, u6_dot)

    # Index        0    3  7    10     13           14
    x    = vertcat(pos, q, vel, omega)
    xdot = vertcat(pos_dot, q_dot, vel_dot, omega_dot)

    nx = x.size()[0]
    weight_diag = np.ones((nx,)) * 1e-6  # default weight
    weight_diag[0] = 10.0 #30000000.0   # pos East
    weight_diag[1] = 10.0   # pos North
    weight_diag[2] = 50.0#50000000000.0   # altitude
    
    weight_diag[3] = 1.0
    weight_diag[4] = 1.0
    weight_diag[5] = 1.0
    weight_diag[6] = 1.0

    weight_diag[7] = 1.0    # East velocity
    weight_diag[8] = 1.0    # North velocity
    weight_diag[9] = 1.0    # vertical velocity
    
    weight_diag[10] = 1.0 
    weight_diag[11] = 1.0 
    weight_diag[12] = 1.0  

    # Control input u
    u = SX.sym('u', 6, 1) # motor rotation rates

    # System Dynamics
    # ---------------
    R_b_to_n = SX.sym('R_b_to_n', 3, 3) # Matrix to transform from body to local coordinate system
    R_b_to_n[0, 0] = 1.0 - 2.0*q_2*q_2 - 2.0*q_3*q_3
    R_b_to_n[0, 1] = 2.0*q_1*q_2 - 2.0*q_3*q_0
    R_b_to_n[0, 2] = 2.0*q_1*q_3 + 2.0*q_2*q_0
    R_b_to_n[1, 0] = 2.0*q_1*q_2 + 2.0*q_3*q_0
    R_b_to_n[1, 1] = 1.0 - 2.0*q_1*q_1 - 2.0*q_3*q_3
    R_b_to_n[1, 2] = 2.0*q_2*q_3 - 2.0*q_1*q_0
    R_b_to_n[2, 0] = 2.0*q_1*q_3 - 2.0*q_2*q_0
    R_b_to_n[2, 1] = 2.0*q_2*q_3 + 2.0*q_1*q_0
    R_b_to_n[2, 2] = 1.0 - 2.0*q_1*q_1 - 2.0*q_2*q_2
    
    thrust_body = SX.sym('thrust_body', 3, 1)
    
    thrust_body = thrust_vectors[:, 0]*u[0] + thrust_vectors[:, 1]*u[1] + thrust_vectors[:, 2]*u[2] + thrust_vectors[:, 3]*u[3] + thrust_vectors[:, 4]*u[4] + thrust_vectors[:, 5]*u[5]
    #thrust_body = thrust_body * 1.0e-8

    gravity_n = SX.sym('gravity_n', 3, 1)
    gravity_n[0] = 0.0
    gravity_n[1] = 0.0
    gravity_n[2] = -gravity

    vel_dot = R_b_to_n@(thrust_body/mass_kg) + gravity_n
    
    omega_dot = J_inv @ (M @ u - cross(omega, (J @ omega)))  #Euler Gyroscopic Equation 
   
    # Core of the MPC magic
    f_expl = vertcat( 	vel,
    			-(omega_x*q_1)/2.0 - (omega_y*q_2)/2.0 - (omega_z*q_3)/2.0,
                       (omega_x*q_0)/2.0 - (omega_y*q_3)/2.0 + (omega_z*q_2)/2.0,
                       (omega_y*q_0)/2.0 + (omega_x*q_3)/2.0 - (omega_z*q_1)/2.0,
                       (omega_y*q_1)/2.0 - (omega_x*q_2)/2.0 + (omega_z*q_0)/2.0,
                       vel_dot,
                       omega_dot
                     )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    model.weight_diag = weight_diag

    return model

