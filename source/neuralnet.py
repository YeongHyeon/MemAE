import tensorflow as tf

class MemAE(object):

    def __init__(self, height, width, channel, learning_rate=1e-3):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel = height, width, channel
        self.alpha, self.learning_rate = 0.0002, learning_rate
        self.training = False

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.batch_size = tf.placeholder(tf.int32, shape=[])

        self.weights, self.biasis = [], []
        self.w_names, self.b_names = [], []
        self.fc_shapes, self.conv_shapes = [], []

        self.x_hat, self.w_hat = self.build_model(input=self.x)

        self.mse_r = self.mean_square_error(x1=self.x, x2=self.x_hat)
        self.mem_etrp = tf.compat.v1.reduce_sum((-self.w_hat) * tf.math.log(self.w_hat + 1e-12), axis=(1, 2, 3))
        self.loss = tf.compat.v1.reduce_mean(self.mse_r + (self.alpha * self.mem_etrp))

        #default: beta1=0.9, beta2=0.999
        self.optimizer = tf.compat.v1.train.AdamOptimizer( \
            self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.loss)

        tf.compat.v1.summary.scalar('MemAE/mse', tf.compat.v1.reduce_sum(self.mse_r))
        tf.compat.v1.summary.scalar('MemAE/w-entropy', tf.compat.v1.reduce_sum(self.mem_etrp))
        tf.compat.v1.summary.scalar('MemAE/total loss', self.loss)
        self.summaries = tf.compat.v1.summary.merge_all()

    def set_training(self): self.training = True

    def set_test(self): self.training = False

    def mean_square_error(self, x1, x2):

        data_dim = len(x1.shape)
        if(data_dim == 4):
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2), axis=(1, 2, 3))
        elif(data_dim == 3):
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2), axis=(1, 2))
        elif(data_dim == 2):
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2), axis=(1))
        else:
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2))

    def cosine_sim(self, x1, x2):

        num = tf.compat.v1.matmul(x1, tf.transpose(x2, perm=[0, 1, 3, 2]), name='attention_num')
        denom =  tf.compat.v1.matmul(x1**2, tf.transpose(x2, perm=[0, 1, 3, 2])**2, name='attention_denum')
        w = (num + 1e-12) / (denom + 1e-12)

        return w

    def build_model(self, input):

        with tf.name_scope('encoder') as scope_enc:
            z_enc, z_c = self.encoder(input=input)

        with tf.name_scope('memory') as scope_enc:
            z_hat, w_hat = self.memory(input=z_enc, n=2000, c=z_c)

        with tf.name_scope('decoder') as scope_enc:
            x_hat = self.decoder(input=z_hat)

        return x_hat, w_hat

    def encoder(self, input):

        print("Encode")
        self.conv_shapes.append(input.shape)

        conv1 = self.conv2d(input=input, stride=2, padding='SAME', \
            filter_size=[1, 1, 1, 16], name="conv1")
        bn1 = self.batch_normalization(input=conv1, name="bn1")
        act1 = tf.compat.v1.nn.relu(bn1)
        self.conv_shapes.append(act1.shape)

        conv2 = self.conv2d(input=act1, stride=2, padding='SAME', \
            filter_size=[3, 3, 16, 32], name="conv2")
        bn2 = self.batch_normalization(input=conv2, name="bn2")
        act2 = tf.compat.v1.nn.relu(bn2)
        self.conv_shapes.append(act2.shape)

        conv3 = self.conv2d(input=act2, stride=2, padding='SAME', \
            filter_size=[3, 3, 32, 64], name="conv3")
        bn3 = self.batch_normalization(input=conv3, name="bn3")
        act3 = tf.compat.v1.nn.relu(bn3)

        [n, h, w, c] = act3.shape
        z = act3

        return z, c

    def memory(self, input, n=2000, c=64):

        # N = Memory Capacity
        self.weights, self.w_names, w_memory = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, shape=[1, 1, n, c], name='w_memory')

        print("Attention for Memory Addressing")
        cosim = self.cosine_sim(x1=input, x2=w_memory) # Eq.5
        atteniton = tf.nn.softmax(cosim)# Eq.4
        print("Attention  z", input.shape, "M_T", tf.transpose(w_memory, perm=[0, 1, 3, 2]).shape, "->", atteniton.shape)

        print("Hard Shrinkage for Sparse Addressing")
        lam = 1 / n # deactivate the 1/N of N memories.

        addr_num = tf.compat.v1.nn.relu(atteniton - lam) * atteniton
        addr_denum = tf.abs(atteniton - lam) + 1e-12
        memory_addr = addr_num / addr_denum

        renorm = tf.compat.v1.clip_by_value(memory_addr, 1e-12, 1-(1e-12))

        z_hat = tf.compat.v1.matmul(renorm, w_memory, name='shrinkage')
        print("Shrinkage  w", renorm.shape, "M", w_memory.shape, "->", z_hat.shape)

        return z_hat, renorm

    def decoder(self, input):

        z_hat = input

        print("Decode")
        [n, h, w, c] = self.conv_shapes[-1]
        convt1 = self.conv2d_transpose(input=z_hat, stride=2, padding='SAME', \
            output_shape=[self.batch_size, h, w, c], filter_size=[3, 3, c, 64], \
            dilations=[1, 1, 1, 1], name="convt1")
        bnt1 = self.batch_normalization(input=convt1, name="bnt1")
        actt1 = tf.compat.v1.nn.relu(bnt1)

        [n, h, w, c] = self.conv_shapes[-2]
        convt2 = self.conv2d_transpose(input=actt1, stride=2, padding='SAME', \
            output_shape=[self.batch_size, h, w, c], filter_size=[3, 3, c, 32], \
            dilations=[1, 1, 1, 1], name="convt2")
        bnt2 = self.batch_normalization(input=convt2, name="bnt2")
        actt2 = tf.compat.v1.nn.relu(bnt2)

        [n, h, w, c] = self.conv_shapes[-3]
        convt3 = self.conv2d_transpose(input=actt2, stride=2, padding='SAME', \
            output_shape=[self.batch_size, h, w, c], filter_size=[3, 3, c, 16], \
            dilations=[1, 1, 1, 1], name="convt3")

        x_hat = tf.compat.v1.clip_by_value(convt3, 1e-12, 1-(1e-12))

        return x_hat

    def initializer(self):
        return tf.compat.v1.initializers.variance_scaling(distribution="untruncated_normal", dtype=tf.dtypes.float32)

    def variable_maker(self, var_bank, name_bank, shape, name=""):

        try:
            var_idx = name_bank.index(name)
        except:
            variable = tf.compat.v1.get_variable(name=name, \
                shape=shape, initializer=self.initializer())

            var_bank.append(variable)
            name_bank.append(name)
        else:
            variable = var_bank[var_idx]

        return var_bank, name_bank, variable

    def batch_normalization(self, input, name=""):

        bnlayer = tf.compat.v1.keras.layers.BatchNormalization(
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer='zeros',
            gamma_initializer='ones',
            moving_mean_initializer='zeros',
            moving_variance_initializer='ones',
            renorm_momentum=0.99,
            trainable=True,
            name="%s_bn" %(name),
        )

        bn = bnlayer(inputs=input, training=self.training)
        return bn

    def conv2d(self, input, stride, padding, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], name=""):

        # strides=[N, H, W, C], [1, stride, stride, 1]
        # filter_size=[ksize, ksize, num_inputs, num_outputs]
        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=filter_size, name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[filter_size[-1]], name='%s_b' %(name))

        out_conv = tf.compat.v1.nn.conv2d(
            input=input,
            filter=weight,
            strides=[1, stride, stride, 1],
            padding=padding,
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv' %(name),
        )
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))

        print("Conv", input.shape, "->", out_bias.shape)
        return out_bias

    def conv2d_transpose(self, input, stride, padding, output_shape, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], name=""):

        # strides=[N, H, W, C], [1, stride, stride, 1]
        # filter_size=[ksize, ksize, num_outputs, num_inputs]
        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=filter_size, name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[filter_size[-2]], name='%s_b' %(name))

        out_conv = tf.compat.v1.nn.conv2d_transpose(
            value=input,
            filter=weight,
            output_shape=output_shape,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv_tr' %(name),
        )
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))

        print("Conv-Tr", input.shape, "->", out_bias.shape)
        return out_bias
