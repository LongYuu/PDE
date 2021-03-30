import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time

# custome neural network
class NeuralNetwork:
    # Initialize the class
    def __init__(self, x, t, u, layers, activation):        
        X = np.concatenate([x, t], 1)

        self.lb = X.min(0)  # uniform
        self.ub = X.max(0)
        
        self.X = X

        self.x = X[:, 0:1]
        self.t = X[:, 1:2]

        self.u = u

        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        self.activation = activation

        # tf Placeholders
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        # tf Graphs
        self.u_pred, self.u_t_pred, self.u_x_pred, self.u_xx_pred, self.u_xxx_pred = self.net_pde(
            self.x_tf, self.t_tf)
       
        # Loss
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) 
        
        # Optimizers
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=1e6)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))  

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def __init__(self, x, y, t, u, layers, activation):        
        X = np.concatenate([x, y, t], 1)

        self.lb = X.min(0)  # uniform
        self.ub = X.max(0)
        
        self.X = X

        self.x = X[:, 0:1]
        self.y = X[:, 1:2]
        self.t = X[:, 2:3]        

        self.u = u

        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        self.activation = activation

        # tf Placeholders
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        # tf Graphs
        self.u_pred, self.u_t_pred, self.u_x_pred, self.u_y_pred, self.u_xx_pred, self.u_xy_pred, self.u_yy_pred, self.u_xxx_pred, self.u_xxy_pred, self.u_xyy_pred, self.u_yyy_pred = self.net_pde(
            self.x_tf, self.y_tf, self.t_tf)
       
        # Loss
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) 
        
        # Optimizers
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))  

        init = tf.global_variables_initializer()
        self.sess.run(init)
    
    
    def __init__(self, x, y, z, t, u, layers, activation):
        X = np.concatenate([x, y, z, t], 1)
        
        self.lb = X.min(0)    #uniform
        self.ub = X.max(0)

        self.X = X 

        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.z = X[:,2:3]
        self.t = X[:,3:4]

        self.u = u
        
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        self.activation = activation
        
        # tf Placeholders        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.z_tf = tf.placeholder(tf.float32, shape=[None, self.z.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        # tf Graphs
        self.u_pred, self.u_t_pred, self.u_x_pred, self.u_y_pred, self.u_z_pred, self.u_xx_pred, self.u_xy_pred, self.u_xz_pred, self.u_yy_pred, self.u_yz_pred, self.u_zz_pred, self.u_xxx_pred, self.u_xxy_pred, self.u_xxz_pred, self.u_xyy_pred, self.u_xyz_pred, self.u_xzz_pred, self.u_yyy_pred, self.u_yyz_pred, self.u_yzz_pred, self.u_zzz_pred = self.net_pde(
            self.x_tf, self.y_tf, self.z_tf, self.t_tf)
        
        # Loss
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        
        # Optimizers                                                                  
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
    
    def initialize_NN(self, layers):        
        weights = []        
        biases = []        
        num_layers = len(layers)
        for l in range(0, num_layers - 1):            
            W = self.xavier_init(size=[layers[l], layers[l + 1]])            
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)            
            weights.append(W)            
            biases.append(b)
        return weights, biases
        
    def xavier_init(self, size):        
        in_dim = size[0]        
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim)) 
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev),dtype=tf.float32)
    
    def neural_net(self, X, weights, biases, activation):        
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0  # uniform
        for l in range(0, num_layers - 2):            
            W = weights[l]            
            b = biases[l]
            if activation == 'sigmoid':            
                H = tf.sigmoid(tf.add(tf.matmul(tf.cast(H, tf.float32), W), b))
            elif activation == 'tanh':
                H = tf.nn.tanh(tf.add(tf.matmul(tf.cast(H, tf.float32), W), b))
            elif activation == 'sin':
                H = tf.sin(tf.add(tf.matmul(tf.cast(H, tf.float32), W), b))
        W = weights[-1]        
        b = biases[-1]        
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_pde(self, x, t):        
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases, self.activation)
        
        # time derivatives
        u_t = tf.gradients(u, t)[0]
        # first derivatives
        u_x = tf.gradients(u, x)[0]
        # second derivatives
        u_xx = tf.gradients(u_x, x)[0]
        # third derivatives
        u_xxx = tf.gradients(u_xx, x)[0]

        return u, u_t, u_x, u_xx, u_xxx
    
    def net_pde(self, x, y, t):        
        u = self.neural_net(tf.concat([x, y, t], 1), self.weights, self.biases, self.activation)
        
        # time derivatives
        u_t = tf.gradients(u, t)[0]
        # first derivatives
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        # second derivatives
        u_xx = tf.gradients(u_x, x)[0]
        u_xy = tf.gradients(u_x, y)[0]
        u_yy = tf.gradients(u_y, y)[0]
        # third derivatives
        u_xxx = tf.gradients(u_xx, x)[0]
        u_xxy = tf.gradients(u_xx, y)[0]
        u_xyy = tf.gradients(u_xy, y)[0]
        u_yyy = tf.gradients(u_yy, y)[0]

        return u, u_t, u_x, u_y, u_xx, u_xy, u_yy, u_xxx, u_xxy, u_xyy, u_yyy
    
    def net_pde(self, x, y, z, t):        
        u = self.neural_net(tf.concat([x, y, z, t], 1), self.weights, self.biases, self.activation)
        
        # time derivatives
        u_t = tf.gradients(u, t)[0]
        # first derivatives
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_z = tf.gradients(u, z)[0]
        # second derivatives
        u_xx = tf.gradients(u_x, x)[0]
        u_xy = tf.gradients(u_x, y)[0]
        u_xz = tf.gradients(u_x, z)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_yz = tf.gradients(u_y, z)[0]
        u_zz = tf.gradients(u_z, z)[0]
        # third derivatives
        u_xxx = tf.gradients(u_xx, x)[0]
        u_xxy = tf.gradients(u_xx, y)[0]
        u_xxz = tf.gradients(u_xx, z)[0]
        u_xyy = tf.gradients(u_xy, y)[0]
        u_xyz = tf.gradients(u_xy, z)[0]
        u_xzz = tf.gradients(u_xz, z)[0]
        u_yyy = tf.gradients(u_yy, y)[0]
        u_yyz = tf.gradients(u_yy, z)[0]
        u_yzz = tf.gradients(u_yz, z)[0]
        u_zzz = tf.gradients(u_zz, z)[0]
    
        return u, u_t, u_x, u_y, u_z, u_xx, u_xy, u_xz, u_yy, u_yz, u_zz, u_xxx, u_xxy, u_xxz, u_xyy, u_xyz, u_xzz, u_yyy, u_yyz, u_yzz, u_zzz
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter, varNum):      
        
        if varNum == 1:
            tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u} 
        elif varNum == 2:
            tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t,
                self.u_tf: self.u}
        elif varNum == 3:
            tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.z_tf: self.z, 
                self.t_tf: self.t, self.u_tf: self.u}

        start_time = time.time()
        loss = []
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:                
                elapsed = time.time() - start_time                
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()

            if it % (nIter // 11) == 0:
                loss.append(loss_value.item())

        self.callback(loss_value)

        return loss
    
    def predict(self, X_star, varNum):
        if varNum == 1:        
            tf_dict = {self.x_tf: X_star[:, 0:1], self.t_tf: X_star[:, 1:2]}

            u_star = self.sess.run(self.u_pred, tf_dict) 
            u_t_star = self.sess.run(self.u_t_pred, tf_dict)       
            u_x_star = self.sess.run(self.u_x_pred, tf_dict)        
            u_xx_star = self.sess.run(self.u_xx_pred, tf_dict)
            u_xxx_star = self.sess.run(self.u_xxx_pred, tf_dict)

            return [u_star, u_t_star, u_x_star, u_xx_star, u_xxx_star]

        elif varNum == 2:
            tf_dict = {self.x_tf: X_star[:, 0:1], self.y_tf: X_star[:, 1:2],
                self.t_tf: X_star[:, 2:3]}

            u_star = self.sess.run(self.u_pred, tf_dict) 
            u_t_star = self.sess.run(self.u_t_pred, tf_dict)       
            u_x_star = self.sess.run(self.u_x_pred, tf_dict)
            u_y_star = self.sess.run(self.u_y_pred, tf_dict)        
            u_xx_star = self.sess.run(self.u_xx_pred, tf_dict)
            u_xy_star = self.sess.run(self.u_xy_pred, tf_dict)
            u_yy_star = self.sess.run(self.u_yy_pred, tf_dict)
            u_xxx_star = self.sess.run(self.u_xxx_pred, tf_dict)
            u_xxy_star = self.sess.run(self.u_xxy_pred, tf_dict)
            u_xyy_star = self.sess.run(self.u_xyy_pred, tf_dict)
            u_yyy_star = self.sess.run(self.u_yyy_pred, tf_dict)

            return [u_star, u_t_star, u_x_star, u_y_star, u_xx_star, u_xy_star, u_yy_star, u_xxx_star, u_xxy_star, u_xyy_star, u_yyy_star]

        elif varNum == 3:
            tf_dict = {self.x_tf: X_star[:, 0:1], self.y_tf: X_star[:, 1:2],
                self.z_tf: X_star[:, 2:3], self.t_tf: X_star[:, 3:4]}
            
            u_star = self.sess.run(self.u_pred, tf_dict) 
            u_t_star = self.sess.run(self.u_t_pred, tf_dict)       
            u_x_star = self.sess.run(self.u_x_pred, tf_dict)
            u_y_star = self.sess.run(self.u_y_pred, tf_dict)
            u_z_star = self.sess.run(self.u_z_pred, tf_dict)           
            u_xx_star = self.sess.run(self.u_xx_pred, tf_dict)
            u_xy_star = self.sess.run(self.u_xy_pred, tf_dict)
            u_xz_star = self.sess.run(self.u_xz_pred, tf_dict)
            u_yy_star = self.sess.run(self.u_yy_pred, tf_dict)
            u_yz_star = self.sess.run(self.u_yz_pred, tf_dict)
            u_zz_star = self.sess.run(self.u_zz_pred, tf_dict)
            u_xxx_star = self.sess.run(self.u_xxx_pred, tf_dict)
            u_xxy_star = self.sess.run(self.u_xxy_pred, tf_dict)
            u_xxz_star = self.sess.run(self.u_xxz_pred, tf_dict)
            u_xyy_star = self.sess.run(self.u_xyy_pred, tf_dict)
            u_xyz_star = self.sess.run(self.u_xyz_pred, tf_dict)
            u_xzz_star = self.sess.run(self.u_xzz_pred, tf_dict)
            u_yyy_star = self.sess.run(self.u_yyy_pred, tf_dict)
            u_yyz_star = self.sess.run(self.u_yyz_pred, tf_dict)
            u_yzz_star = self.sess.run(self.u_yzz_pred, tf_dict)
            u_zzz_star = self.sess.run(self.u_zzz_pred, tf_dict)

            return [u_star, u_t_star, u_x_star, u_y_star, u_z_star, u_xx_star, u_xy_star, u_xz_star, u_yy_star, u_yz_star, u_zz_star, u_xxx_star, u_xxy_star, u_xxz_star, u_xyy_star, u_xyz_star, u_xzz_star, u_yyy_star, u_yyz_star, u_yzz_star, u_zzz_star]