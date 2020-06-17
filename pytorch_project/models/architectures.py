import torch
import torch.nn as nn
from torch.autograd import Variable

class NetTest(nn.Module):
	def __init__(self, input_size, use_gpu=True):
		super(NetTest, self).__init__()
		if use_gpu:
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
		w_conv1 = Variable(torch.randn((16, 1, 5, 5, 5), device=device)*0.01, requires_grad=True)
		w_conv2 = Variable(torch.randn((32, 16, 3, 3, 3), device=device)*0.01, requires_grad=True)
		w_fc1 = Variable(torch.randn((2, int(input_size/4)**3*32), device=device)*0.01, requires_grad=True)
		self.params = nn.ParameterDict({'w_conv1': nn.Parameter(w_conv1), 'w_conv2': nn.Parameter(w_conv2), 'w_fc1': nn.Parameter(w_fc1)})

	def forward(self, input, fin_feature=None):
		out_conv1 = nn.functional.conv3d(input, self.params['w_conv1'], padding=2)
		out_rl1 = nn.functional.relu(out_conv1)
		out_pool1 = nn.functional.max_pool3d(out_rl1, 2, 2)

		out_conv2 = nn.functional.conv3d(out_pool1, self.params['w_conv2'], padding=1)
		out_rl2 = nn.functional.relu(out_conv2)
		out_pool2 = nn.functional.max_pool3d(out_rl2, 2, 2)

		feature_flattened = out_pool2.view(out_pool2.size(0), -1)
		output = nn.functional.linear(feature_flattened, self.params['w_fc1'])

		if fin_feature is None:
			return output
		else:
			return output, feature_flattened

	#def parameters(self):
	#	return [self.w_conv1, self.w_conv2, self.w_fc1]
	
	def save(self, str1, str2):
		print(str1+str2)

class ArchTest(nn.Module):
	def __init__(self, input_size, use_gpu=True):
		super(ArchTest, self).__init__()
		if use_gpu:
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
		self.convtest = nn.Conv3d(1, 1, 1)
		self.m_fc = Variable(torch.randn((2, input_size**3), device=device)*0.01, requires_grad=True)

	def forward(self, input, fin_feature=None):
		feature_flattened = input.view(input.size(0), -1)
		output = nn.functional.linear(feature_flattened, self.m_fc)

		if fin_feature is None:
			return output
		else:
			return output, feature_flattened

	def save(self, str1, str2):
		print(str1+str2)
'''
def volume_bndo_flbias_l6_40_v2(input, training=True, positive_confidence=0.5, dropout_rate=0.3, batch_normalization_statistic=True, bn_params=None):
	#batch normalization net for focal loss training with bias fine tuned for positive_confidence
	#dropout layer adapted
	bn_variance_epsilon = tf.constant(0.0000000000001, dtype=tf.float32)
	keep_prob = tf.constant(1-dropout_rate, dtype=tf.float32)

	w_conv1 = tf.Variable(tf.random_normal([5, 5, 5, 1, 64], stddev=0.1), dtype=tf.float32, name='w_conv1', trainable=training)
	out_conv1 = tf.nn.conv3d(input, w_conv1, strides=[1, 1, 1, 1, 1], padding='VALID')
	if bn_params is None:
		bn_mean1, bn_var1 = tf.nn.moments(out_conv1, [i for i in range(len(out_conv1.shape))])
	else:
		bn_mean1 = tf.constant(bn_params[0][0], dtype=tf.float32, name='bn_mean1')
		bn_var1 = tf.constant(bn_params[0][1], dtype=tf.float32, name='bn_var1')
	r_bn1 = tf.Variable([1], dtype=tf.float32, name='r_bn1', trainable=training)
	b_bn1 = tf.Variable(tf.random_normal([64], stddev=0.1), dtype=tf.float32, name='b_bn1', trainable=training)
	out_bn1 = tf.nn.batch_normalization(out_conv1, bn_mean1, bn_var1, b_bn1, r_bn1, bn_variance_epsilon)
	out_dropout1 = tf.nn.dropout(tf.nn.relu(out_bn1), keep_prob)
	hidden_conv1 = tf.nn.max_pool3d(out_dropout1, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv1 ,the output volume size is 18x18x18([batch_size,in_deep,width,height,output_deep])
	w_conv2 = tf.Variable(tf.random_normal([5, 5, 5, 64, 128], stddev=0.1), dtype=tf.float32, name='w_conv2', trainable=training)
	out_conv2 = tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1, 1, 1], padding='VALID')
	if bn_params is None:
		bn_mean2, bn_var2 = tf.nn.moments(out_conv2, [i for i in range(len(out_conv2.shape))])
	else:
		bn_mean2 = tf.constant(bn_params[1][0], dtype=tf.float32, name='bn_mean2')
		bn_var2 = tf.constant(bn_params[1][1], dtype=tf.float32, name='bn_var2')
	r_bn2 = tf.Variable([1], dtype=tf.float32, name='r_bn2', trainable=training)
	b_bn2 = tf.Variable(tf.random_normal([128], stddev=0.1), dtype=tf.float32, name='b_bn2', trainable=training)
	out_bn2 = tf.nn.batch_normalization(out_conv2, bn_mean2, bn_var2, b_bn2, r_bn2, bn_variance_epsilon)
	out_dropout2 = tf.nn.dropout(tf.nn.relu(out_bn2), keep_prob)
	hidden_conv2 = tf.nn.max_pool3d(out_dropout2, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv2 ,the output volume size is 7x7x7([batch_size,in_deep,width,height,output_deep])
	w_conv3 = tf.Variable(tf.random_normal([3, 3, 3, 128, 256], stddev=0.1), dtype=tf.float32, name='w_conv3', trainable=training)
	out_conv3 = tf.nn.conv3d(hidden_conv2, w_conv3, strides=[1, 1, 1, 1, 1], padding='VALID')
	if bn_params is None:
		bn_mean3, bn_var3 = tf.nn.moments(out_conv3, [i for i in range(len(out_conv3.shape))])
	else:
		bn_mean3 = tf.constant(bn_params[2][0], dtype=tf.float32, name='bn_mean3')
		bn_var3 = tf.constant(bn_params[2][1], dtype=tf.float32, name='bn_var3')
	r_bn3 = tf.Variable([1], dtype=tf.float32, name='r_bn3', trainable=training)
	b_bn3 = tf.Variable(tf.random_normal([256], stddev=0.1), dtype=tf.float32, name='b_bn3', trainable=training)
	out_bn3 = tf.nn.batch_normalization(out_conv3, bn_mean3, bn_var3, b_bn3, r_bn3, bn_variance_epsilon)
	out_dropout3 = tf.nn.dropout(tf.nn.relu(out_bn3), keep_prob)

	# after conv3 ,the output volume size is 5x5x5([batch_size,in_deep,width,height,output_deep])
	w_conv4 = tf.Variable(tf.random_normal([5, 5, 5, 256, 512], stddev=0.1), dtype=tf.float32, name='w_conv4', trainable=training)
	out_conv4 = tf.nn.conv3d(out_dropout3, w_conv4, strides=[1, 1, 1, 1, 1], padding='VALID')
	if bn_params is None:
		bn_mean4, bn_var4 = tf.nn.moments(out_conv4, [i for i in range(len(out_conv4.shape))])
	else:
		bn_mean4 = tf.constant(bn_params[3][0], dtype=tf.float32, name='bn_mean4')
		bn_var4 = tf.constant(bn_params[3][1], dtype=tf.float32, name='bn_var4')
	r_bn4 = tf.Variable([1], dtype=tf.float32, name='r_bn4', trainable=training)
	b_bn4 = tf.Variable(tf.random_normal([512], stddev=0.1), dtype=tf.float32, name='b_bn4', trainable=training)
	out_bn4 = tf.nn.batch_normalization(out_conv4, bn_mean4, bn_var4, b_bn4, r_bn4, bn_variance_epsilon)
	out_dropout4 = tf.nn.dropout(tf.nn.relu(out_bn4), keep_prob)

	# after conv3 ,the output size is batch_sizex1x1x1([batch_size,in_deep,width,height,output_deep])
	# all feature map flatten to one dimension vector,this vector will be much long
	flattened_conv4 = tf.reshape(out_dropout4, [-1, 512])
	w_fc1 = tf.Variable(tf.random_normal([512, 512], stddev=0.1), name='w_fc1', trainable=training)
	out_fc1 = tf.matmul(flattened_conv4, w_fc1)
	r_bn5 = tf.Variable([1], dtype=tf.float32, name='r_bn5', trainable=training)
	b_bn5 = tf.Variable(tf.random_normal([512], stddev=0.1), dtype=tf.float32, name='b_bn5', trainable=training)
	if bn_params is None:
		bn_mean5, bn_var5 = tf.nn.moments(out_fc1, [i for i in range(len(out_fc1.shape))])
	else:
		bn_mean5 = tf.constant(bn_params[4][0], dtype=tf.float32, name='bn_mean5')
		bn_var5 = tf.constant(bn_params[4][1], dtype=tf.float32, name='bn_var5')
	out_bn5 = tf.nn.batch_normalization(out_fc1, bn_mean5, bn_var5, b_bn5, r_bn5, bn_variance_epsilon)
	out_dropout5 = tf.nn.dropout(tf.nn.relu(out_bn5), keep_prob)

	w_fc2 = tf.Variable(tf.random_normal([512, 2], stddev=0.1), name='w_fc2', trainable=training)
	#b_fc2 = tf.Variable(tf.random_normal(shape=[2], mean=0, stddev=0.1) + tf.constant([-math.log((1-positive_confidence)/positive_confidence), math.log((1-positive_confidence)/positive_confidence)]), name='b_fc2', trainable=training)
	b_fc2 = tf.Variable([-math.log((1-positive_confidence)/positive_confidence), math.log((1-positive_confidence)/positive_confidence)], name='b_fc2', trainable=training)
	out_fc2 = tf.add(tf.matmul(out_dropout5, w_fc2), b_fc2)

	out_sm = tf.nn.softmax(out_fc2)
	
	outputs = {'conv1_out':hidden_conv1, 'conv2_out':hidden_conv2, 'conv3_out':out_dropout3, 'conv4_out':out_dropout4, 'flattened_out':flattened_conv4, 'fc1_out':out_dropout5, 'last_out':out_fc2, 'sm_out':out_sm}
	variables = {'w_conv1':w_conv1, 'r_bn1':r_bn1, 'b_bn1':b_bn1,
		     'w_conv2':w_conv2, 'r_bn2':r_bn2, 'b_bn2':b_bn2,
		     'w_conv3':w_conv3, 'r_bn3':r_bn3, 'b_bn3':b_bn3,
		     'w_conv4':w_conv4, 'r_bn4':r_bn4, 'b_bn4':b_bn4,
		     'w_fc1':w_fc1, 'r_bn5':r_bn5, 'b_bn5':b_bn5,
		     'w_fc2':w_fc2, 'b_fc2':b_fc2}
	
	if batch_normalization_statistic:
		bn_pars = []
		bn_pars.append([bn_mean1, bn_var1])
		bn_pars.append([bn_mean2, bn_var2])
		bn_pars.append([bn_mean3, bn_var3])
		bn_pars.append([bn_mean4, bn_var4])
		bn_pars.append([bn_mean5, bn_var5])
	else:
		bn_pars = None

	return outputs, variables, bn_pars
'''
