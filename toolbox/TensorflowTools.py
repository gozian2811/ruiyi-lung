import math
import tensorflow as tf

def mlc_archi_3(input, keep_prob):
	with tf.name_scope("Archi-3"):
		# input size is batch_sizex20x20x6
		# 5x5x3 is the kernel size of conv1,1 is the input depth,64 is the number output channel
		w_conv1 = tf.Variable(tf.random_normal([3,5,5,1,64],stddev=0.001),dtype=tf.float32,name='w_conv1')
		b_conv1 = tf.Variable(tf.constant(0.01,shape=[64]),dtype=tf.float32,name='b_conv1')
		out_conv1 = tf.nn.relu(tf.add(tf.nn.conv3d(input,w_conv1,strides=[1,1,1,1,1],padding='VALID'),b_conv1))
		out_conv1 = tf.nn.dropout(out_conv1,keep_prob)

		# max pooling ,pooling layer has no effect on the data size
		hidden_conv1 = tf.nn.max_pool3d(out_conv1,strides=[1,1,1,1,1],ksize=[1,2,2,2,1],padding='SAME')

		# after conv1 ,the output size is batch_sizex4x16x16x64([batch_size,in_deep,width,height,output_deep])
		w_conv2 = tf.Variable(tf.random_normal([3,5, 5, 64,64], stddev=0.001), dtype=tf.float32,name='w_conv2')
		b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv2')
		out_conv2 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1,1, 1], padding='VALID'), b_conv2))
		out_conv2 = tf.nn.dropout(out_conv2, keep_prob)

		# after conv2 ,the output size is batch_sizex2x12x12x64([batch_size,in_deep,width,height,output_deep])
		w_conv3 = tf.Variable(tf.random_normal([3,5, 5, 64,64], stddev=0.001), dtype=tf.float32,
					name='w_conv3')
		b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv3')
		out_conv3 = tf.nn.relu(tf.add(tf.nn.conv3d(out_conv2, w_conv3, strides=[1, 1, 1, 1,1], padding='VALID'),b_conv3))
		out_conv3 = tf.nn.dropout(out_conv3, keep_prob)

		out_conv3_shape = tf.shape(out_conv3)
		tf.summary.scalar('out_conv3_shape', out_conv3_shape[0])
		#print(out_conv3)
		# after conv2 ,the output size is batch_sizex2x8x8x64([batch_size,in_deep,width,height,output_deep])
		# all feature map flatten to one dimension vector,this vector will be much long
		out_conv3 = tf.reshape(out_conv3,[-1,64*28*28*20])
		w_fc1 = tf.Variable(tf.random_normal([64*28*28*20,250],stddev=0.001),name='w_fc1')
		out_fc1 = tf.nn.relu(tf.add(tf.matmul(out_conv3,w_fc1),tf.constant(0.001,shape=[250])))
		out_fc1 = tf.nn.dropout(out_fc1,keep_prob)

		out_fc1_shape = tf.shape(out_fc1)
		tf.summary.scalar('out_fc1_shape', out_fc1_shape[0])

		w_fc2 = tf.Variable(tf.random_normal([250, 2], stddev=0.001), name='w_fc2')
		out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1, w_fc2), tf.constant(0.001, shape=[2])))
		out_fc2 = tf.nn.dropout(out_fc2, keep_prob)

		w_sm = tf.Variable(tf.random_normal([2, 2], stddev=0.001), name='w_sm')
		b_sm = tf.constant(0.001, shape=[2])
		out_sm = tf.nn.softmax(tf.add(tf.matmul(out_fc2, w_sm), b_sm))

		return out_sm

def volume_net_l5_56(input, dropout_rate=0.3):
	keep_prob = tf.constant(1-dropout_rate, dtype=tf.float32)
	w_conv1 = tf.Variable(tf.random_normal([3,3,3,1,16],stddev=0.1),dtype=tf.float32,name='w_conv1')
	b_conv1 = tf.Variable(tf.random_normal(shape=[16], stddev=0.1),dtype=tf.float32,name='b_conv1')
	out_conv1 = tf.nn.relu(tf.add(tf.nn.conv3d(input,w_conv1,strides=[1,1,1,1,1],padding='VALID'),b_conv1))
	dropout_conv1 = tf.nn.dropout(out_conv1,keep_prob)
	hidden_conv1 = tf.nn.max_pool3d(dropout_conv1,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv1 ,the output size is batch_sizex27x27x27x16([batch_size,in_deep,width,height,output_deep])
	w_conv2 = tf.Variable(tf.random_normal([4,4,4,16,32], stddev=0.1), dtype=tf.float32,name='w_conv2')
	b_conv2 = tf.Variable(tf.random_normal(shape=[32], stddev=0.1), dtype=tf.float32, name='b_conv2')
	out_conv2 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1,1,1,1,1], padding='VALID'), b_conv2))
	dropout_conv2 = tf.nn.dropout(out_conv2, keep_prob)
	hidden_conv2 = tf.nn.max_pool3d(dropout_conv2,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv2 ,the output size is batch_sizex12x12x12x32([batch_size,in_deep,width,height,output_deep])
	w_conv3 = tf.Variable(tf.random_normal([5,5,5,32,64], stddev=0.1), dtype=tf.float32, name='w_conv3')
	b_conv3 = tf.Variable(tf.random_normal(shape=[64], stddev=0.1), dtype=tf.float32, name='b_conv3')
	out_conv3 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv2, w_conv3, strides=[1,1,1,1,1], padding='VALID'), b_conv3))
	dropout_conv3 = tf.nn.dropout(out_conv3, keep_prob)
	hidden_conv3 = tf.nn.max_pool3d(dropout_conv3,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv3 ,the output size is batch_sizex4x4x4x64([batch_size,in_deep,width,height,output_deep])
	# all feature map flatten to one dimension vector,this vector will be much long
	flattened_conv3 = tf.reshape(hidden_conv3,[-1,64*4*4*4])
	w_fc1 = tf.Variable(tf.random_normal([64*4*4*4,128],stddev=0.1),name='w_fc1')
	b_fc1 = tf.Variable(tf.random_normal(shape=[128], stddev=0.1), name='b_fc1')
	out_fc1 = tf.nn.relu(tf.add(tf.matmul(flattened_conv3,w_fc1),b_fc1))
	dropout_fc1 = tf.nn.dropout(out_fc1,keep_prob)

	w_fc2 = tf.Variable(tf.random_normal([128, 2], stddev=0.1), name='w_fc2')
	b_fc2 = tf.Variable(tf.random_normal(shape=[2], stddev=0.1), name='b_fc2')
	out_fc2 = tf.add(tf.matmul(out_fc1, w_fc2), b_fc2)

	out_sm = tf.nn.softmax(out_fc2)

	#return hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, dropout_fc1, w_fc2, b_fc2, out_fc2, out_sm
	return out_fc2, out_sm


def volume_net2_l5_56(input, dropout_rate=0.3):
	keep_prob = tf.constant(1-dropout_rate, dtype=tf.float32)
	w_conv1 = tf.Variable(tf.random_normal([3,3,3,1,16],stddev=0.1),dtype=tf.float32,name='w_conv1')
	b_conv1 = tf.Variable(tf.random_normal(shape=[16], stddev=0.1),dtype=tf.float32,name='b_conv1')
	out_conv1 = tf.nn.relu(tf.add(tf.nn.conv3d(input,w_conv1,strides=[1,1,1,1,1],padding='VALID'),b_conv1))
	dropout_conv1 = tf.nn.dropout(out_conv1,keep_prob)
	hidden_conv1 = tf.nn.max_pool3d(dropout_conv1,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv1 ,the output size is batch_sizex27x27x27x16([batch_size,in_deep,width,height,output_deep])
	w_conv2 = tf.Variable(tf.random_normal([4,4,4,16,32], stddev=0.1), dtype=tf.float32,name='w_conv2')
	b_conv2 = tf.Variable(tf.random_normal(shape=[32], stddev=0.1), dtype=tf.float32, name='b_conv2')
	out_conv2 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1,1,1,1,1], padding='VALID'), b_conv2))
	dropout_conv2 = tf.nn.dropout(out_conv2, keep_prob)
	hidden_conv2 = tf.nn.max_pool3d(dropout_conv2,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv2 ,the output size is batch_sizex12x12x12x32([batch_size,in_deep,width,height,output_deep])
	w_conv3 = tf.Variable(tf.random_normal([5,5,5,32,64], stddev=0.1), dtype=tf.float32, name='w_conv3')
	b_conv3 = tf.Variable(tf.random_normal(shape=[64], stddev=0.1), dtype=tf.float32, name='b_conv3')
	out_conv3 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv2, w_conv3, strides=[1,1,1,1,1], padding='VALID'), b_conv3))
	dropout_conv3 = tf.nn.dropout(out_conv3, keep_prob)
	hidden_conv3 = tf.nn.max_pool3d(dropout_conv3,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv3 ,the output size is batch_sizex5x5x5x64([batch_size,in_deep,width,height,output_deep])
	w_conv4 = tf.Variable(tf.random_normal([4,4,4,64,128], stddev=0.1), dtype=tf.float32, name='w_conv4')
	b_conv4 = tf.Variable(tf.random_normal(shape=[128], stddev=0.1), dtype=tf.float32, name='b_conv4')
	out_conv4 = tf.nn.relu(
		tf.add(tf.nn.conv3d(hidden_conv3, w_conv4, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv4))
	dropout_conv4 = tf.nn.dropout(out_conv4, keep_prob)
	#hidden_conv4 = tf.nn.max_pool3d(dropout_conv4, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv3 ,the output size is batch_sizex4x4x4x128([batch_size,in_deep,width,height,output_deep])
	# all feature map flatten to one dimension vector,this vector will be much long
	flattened_conv4 = tf.reshape(dropout_conv4,[-1,128])
	w_fc1 = tf.Variable(tf.random_normal([128,2],stddev=0.1),name='w_fc1')
	b_fc1 = tf.Variable(tf.random_normal(shape=[2], stddev=0.1), name='b_fc1')
	out_fc1 = tf.nn.relu(tf.add(tf.matmul(flattened_conv4,w_fc1),b_fc1))

	out_sm = tf.nn.softmax(out_fc1)

	#return hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, dropout_fc1, w_fc2, b_fc2, out_fc2, out_sm
	return out_fc1, out_sm

def volume_net_l6_56(input, dropout_rate=0.3):
	keep_prob = tf.constant(1-dropout_rate, dtype=tf.float32)
	w_conv1 = tf.Variable(tf.random_normal([3,3,3,1,16],stddev=0.1),dtype=tf.float32,name='w_conv1')
	b_conv1 = tf.Variable(tf.random_normal(shape=[16], stddev=0.1),dtype=tf.float32,name='b_conv1')
	out_conv1 = tf.nn.relu(tf.add(tf.nn.conv3d(input,w_conv1,strides=[1,1,1,1,1],padding='VALID'),b_conv1))
	dropout_conv1 = tf.nn.dropout(out_conv1,keep_prob)
	hidden_conv1 = tf.nn.max_pool3d(dropout_conv1,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv1 ,the output size is batch_sizex27x27x27x16([batch_size,in_deep,width,height,output_deep])
	w_conv2 = tf.Variable(tf.random_normal([4,4,4,16,32], stddev=0.1), dtype=tf.float32,name='w_conv2')
	b_conv2 = tf.Variable(tf.random_normal(shape=[32], stddev=0.1), dtype=tf.float32, name='b_conv2')
	out_conv2 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1,1,1,1,1], padding='VALID'), b_conv2))
	dropout_conv2 = tf.nn.dropout(out_conv2, keep_prob)
	hidden_conv2 = tf.nn.max_pool3d(dropout_conv2,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv2 ,the output size is batch_sizex12x12x12x32([batch_size,in_deep,width,height,output_deep])
	w_conv3 = tf.Variable(tf.random_normal([5,5,5,32,64], stddev=0.1), dtype=tf.float32, name='w_conv3')
	b_conv3 = tf.Variable(tf.random_normal(shape=[64], stddev=0.1), dtype=tf.float32, name='b_conv3')
	out_conv3 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv2, w_conv3, strides=[1,1,1,1,1], padding='VALID'), b_conv3))
	dropout_conv3 = tf.nn.dropout(out_conv3, keep_prob)
	hidden_conv3 = tf.nn.max_pool3d(dropout_conv3,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv3 ,the output size is batch_sizex4x4x4x64([batch_size,in_deep,width,height,output_deep])
	w_conv4 = tf.Variable(tf.random_normal([4,4,4,64,128], stddev=0.1), dtype=tf.float32, name='w_conv4')
	b_conv4 = tf.Variable(tf.random_normal(shape=[128], stddev=0.1), dtype=tf.float32, name='b_conv4')
	out_conv4 = tf.nn.relu(
		tf.add(tf.nn.conv3d(hidden_conv3, w_conv4, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv4))
	dropout_conv4 = tf.nn.dropout(out_conv4, keep_prob)
	#hidden_conv4 = tf.nn.max_pool3d(dropout_conv4, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv3 ,the output size is batch_sizex4x4x4x128([batch_size,in_deep,width,height,output_deep])
	# all feature map flatten to one dimension vector,this vector will be much long
	flattened_conv4 = tf.reshape(dropout_conv4,[-1,128])
	w_fc1 = tf.Variable(tf.random_normal([128,128],stddev=0.1),name='w_fc1')
	b_fc1 = tf.Variable(tf.random_normal(shape=[128], stddev=0.1), name='b_fc1')
	out_fc1 = tf.nn.relu(tf.add(tf.matmul(flattened_conv4,w_fc1),b_fc1))
	#dropout_fc1 = tf.nn.dropout(out_fc1,keep_prob)

	w_fc2 = tf.Variable(tf.random_normal([128, 2], stddev=0.1), name='w_fc2')
	b_fc2 = tf.Variable(tf.random_normal(shape=[2], stddev=0.1), name='b_fc2')
	out_fc2 = tf.add(tf.matmul(out_fc1, w_fc2), b_fc2)

	out_sm = tf.nn.softmax(out_fc2)

	#return hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, dropout_fc1, w_fc2, b_fc2, out_fc2, out_sm
	return out_fc2, out_sm

def volume_net2_l6_56(input, dropout_rate=0.3):
	keep_prob = tf.constant(1-dropout_rate, dtype=tf.float32)
	w_conv1 = tf.Variable(tf.random_normal([3,3,3,1,16],stddev=0.1),dtype=tf.float32,name='w_conv1')
	b_conv1 = tf.Variable(tf.random_normal(shape=[16], stddev=0.1),dtype=tf.float32,name='b_conv1')
	out_conv1 = tf.nn.relu(tf.add(tf.nn.conv3d(input,w_conv1,strides=[1,1,1,1,1],padding='VALID'),b_conv1))
	dropout_conv1 = tf.nn.dropout(out_conv1,keep_prob)
	hidden_conv1 = tf.nn.max_pool3d(dropout_conv1,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv1 ,the output size is batch_sizex27x27x27x16([batch_size,in_deep,width,height,output_deep])
	w_conv2 = tf.Variable(tf.random_normal([4,4,4,16,32], stddev=0.1), dtype=tf.float32,name='w_conv2')
	b_conv2 = tf.Variable(tf.random_normal(shape=[32], stddev=0.1), dtype=tf.float32, name='b_conv2')
	out_conv2 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1,1,1,1,1], padding='VALID'), b_conv2))
	dropout_conv2 = tf.nn.dropout(out_conv2, keep_prob)
	hidden_conv2 = tf.nn.max_pool3d(dropout_conv2,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv2 ,the output size is batch_sizex12x12x12x32([batch_size,in_deep,width,height,output_deep])
	w_conv3 = tf.Variable(tf.random_normal([3,3,3,32,64], stddev=0.1), dtype=tf.float32, name='w_conv3')
	b_conv3 = tf.Variable(tf.random_normal(shape=[64], stddev=0.1), dtype=tf.float32, name='b_conv3')
	out_conv3 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv2, w_conv3, strides=[1,1,1,1,1], padding='VALID'), b_conv3))
	dropout_conv3 = tf.nn.dropout(out_conv3, keep_prob)
	hidden_conv3 = tf.nn.max_pool3d(dropout_conv3,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv3 ,the output size is batch_sizex4x4x4x64([batch_size,in_deep,width,height,output_deep])
	w_conv4 = tf.Variable(tf.random_normal([5,5,5,64,128], stddev=0.1), dtype=tf.float32, name='w_conv4')
	b_conv4 = tf.Variable(tf.random_normal(shape=[128], stddev=0.1), dtype=tf.float32, name='b_conv4')
	out_conv4 = tf.nn.relu(
		tf.add(tf.nn.conv3d(hidden_conv3, w_conv4, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv4))
	dropout_conv4 = tf.nn.dropout(out_conv4, keep_prob)
	#hidden_conv4 = tf.nn.max_pool3d(dropout_conv4, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv3 ,the output size is batch_sizex4x4x4x128([batch_size,in_deep,width,height,output_deep])
	# all feature map flatten to one dimension vector,this vector will be much long
	flattened_conv4 = tf.reshape(dropout_conv4,[-1,128])
	w_fc1 = tf.Variable(tf.random_normal([128,128],stddev=0.1),name='w_fc1')
	b_fc1 = tf.Variable(tf.random_normal(shape=[128], stddev=0.1), name='b_fc1')
	out_fc1 = tf.nn.relu(tf.add(tf.matmul(flattened_conv4,w_fc1),b_fc1))
	#dropout_fc1 = tf.nn.dropout(out_fc1,keep_prob)

	w_fc2 = tf.Variable(tf.random_normal([128, 2], stddev=0.1), name='w_fc2')
	b_fc2 = tf.Variable(tf.random_normal(shape=[2], stddev=0.1), name='b_fc2')
	out_fc2 = tf.add(tf.matmul(out_fc1, w_fc2), b_fc2)

	out_sm = tf.nn.softmax(out_fc2)

	#return hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, dropout_fc1, w_fc2, b_fc2, out_fc2, out_sm
	return out_fc2, out_sm


def volume_net3_l6_56(input, dropout_rate=0.3):
	keep_prob = tf.constant(1-dropout_rate, dtype=tf.float32)
	w_conv1 = tf.Variable(tf.random_normal([3,3,3,1,64],stddev=0.1),dtype=tf.float32,name='w_conv1')
	b_conv1 = tf.Variable(tf.random_normal(shape=[64], stddev=0.1),dtype=tf.float32,name='b_conv1')
	out_conv1 = tf.nn.relu(tf.add(tf.nn.conv3d(input,w_conv1,strides=[1,1,1,1,1],padding='VALID'),b_conv1))
	dropout_conv1 = tf.nn.dropout(out_conv1,keep_prob)
	hidden_conv1 = tf.nn.max_pool3d(dropout_conv1,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv1 ,the output size is batch_sizex27x27x27x16([batch_size,in_deep,width,height,output_deep])
	w_conv2 = tf.Variable(tf.random_normal([4,4,4,64,128], stddev=0.1), dtype=tf.float32,name='w_conv2')
	b_conv2 = tf.Variable(tf.random_normal(shape=[128], stddev=0.1), dtype=tf.float32, name='b_conv2')
	out_conv2 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1,1,1,1,1], padding='VALID'), b_conv2))
	dropout_conv2 = tf.nn.dropout(out_conv2, keep_prob)
	hidden_conv2 = tf.nn.max_pool3d(dropout_conv2,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv2 ,the output size is batch_sizex12x12x12x32([batch_size,in_deep,width,height,output_deep])
	w_conv3 = tf.Variable(tf.random_normal([3,3,3,128,256], stddev=0.1), dtype=tf.float32, name='w_conv3')
	b_conv3 = tf.Variable(tf.random_normal(shape=[256], stddev=0.1), dtype=tf.float32, name='b_conv3')
	out_conv3 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv2, w_conv3, strides=[1,1,1,1,1], padding='VALID'), b_conv3))
	dropout_conv3 = tf.nn.dropout(out_conv3, keep_prob)
	hidden_conv3 = tf.nn.max_pool3d(dropout_conv3,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv3 ,the output size is batch_sizex5x5x5x64([batch_size,in_deep,width,height,output_deep])
	w_conv4 = tf.Variable(tf.random_normal([5,5,5,256,512], stddev=0.1), dtype=tf.float32, name='w_conv4')
	b_conv4 = tf.Variable(tf.random_normal(shape=[512], stddev=0.1), dtype=tf.float32, name='b_conv4')
	out_conv4 = tf.nn.relu(
		tf.add(tf.nn.conv3d(hidden_conv3, w_conv4, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv4))
	dropout_conv4 = tf.nn.dropout(out_conv4, keep_prob)
	#hidden_conv4 = tf.nn.max_pool3d(dropout_conv4, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv3 ,the output size is batch_sizex4x4x4x128([batch_size,in_deep,width,height,output_deep])
	# all feature map flatten to one dimension vector,this vector will be much long
	flattened_conv4 = tf.reshape(dropout_conv4,[-1,512])
	w_fc1 = tf.Variable(tf.random_normal([512,512],stddev=0.1),name='w_fc1')
	b_fc1 = tf.Variable(tf.random_normal(shape=[512], stddev=0.1), name='b_fc1')
	out_fc1 = tf.nn.relu(tf.add(tf.matmul(flattened_conv4,w_fc1),b_fc1))
	dropout_fc1 = tf.nn.dropout(out_fc1,keep_prob)

	w_fc2 = tf.Variable(tf.random_normal([512, 2], stddev=0.1), name='w_fc2')
	b_fc2 = tf.Variable(tf.random_normal(shape=[2], stddev=0.1), name='b_fc2')
	out_fc2 = tf.add(tf.matmul(out_fc1, w_fc2), b_fc2)

	out_sm = tf.nn.softmax(out_fc2)

	#return hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, dropout_fc1, w_fc2, b_fc2, out_fc2, out_sm
	return out_fc2, out_sm

def volume_net4_l6_56(input, dropout_rate=0.3):
	keep_prob = tf.constant(1-dropout_rate, dtype=tf.float32)
	w_conv1 = tf.Variable(tf.random_normal([3,3,3,1,64],stddev=0.1),dtype=tf.float32,name='w_conv1')
	b_conv1 = tf.Variable(tf.random_normal(shape=[64], stddev=0.1),dtype=tf.float32,name='b_conv1')
	out_conv1 = tf.nn.relu(tf.add(tf.nn.conv3d(input,w_conv1,strides=[1,1,1,1,1],padding='VALID'),b_conv1))
	dropout_conv1 = tf.nn.dropout(out_conv1,keep_prob)
	hidden_conv1 = tf.nn.max_pool3d(dropout_conv1,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv1 ,the output size is batch_sizex27x27x27x16([batch_size,in_deep,width,height,output_deep])
	w_conv2 = tf.Variable(tf.random_normal([4,4,4,64,128], stddev=0.1), dtype=tf.float32,name='w_conv2')
	b_conv2 = tf.Variable(tf.random_normal(shape=[128], stddev=0.1), dtype=tf.float32, name='b_conv2')
	out_conv2 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1,1,1,1,1], padding='VALID'), b_conv2))
	dropout_conv2 = tf.nn.dropout(out_conv2, keep_prob)
	hidden_conv2 = tf.nn.max_pool3d(dropout_conv2,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv2 ,the output size is batch_sizex12x12x12x32([batch_size,in_deep,width,height,output_deep])
	w_conv3 = tf.Variable(tf.random_normal([5,5,5,128,256], stddev=0.1), dtype=tf.float32, name='w_conv3')
	b_conv3 = tf.Variable(tf.random_normal(shape=[256], stddev=0.1), dtype=tf.float32, name='b_conv3')
	out_conv3 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv2, w_conv3, strides=[1,1,1,1,1], padding='VALID'), b_conv3))
	dropout_conv3 = tf.nn.dropout(out_conv3, keep_prob)
	hidden_conv3 = tf.nn.max_pool3d(dropout_conv3,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

	# after conv3 ,the output size is batch_sizex4x4x4x64([batch_size,in_deep,width,height,output_deep])
	w_conv4 = tf.Variable(tf.random_normal([4,4,4,256,512], stddev=0.1), dtype=tf.float32, name='w_conv4')
	b_conv4 = tf.Variable(tf.random_normal(shape=[512], stddev=0.1), dtype=tf.float32, name='b_conv4')
	out_conv4 = tf.nn.relu(
		tf.add(tf.nn.conv3d(hidden_conv3, w_conv4, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv4))
	dropout_conv4 = tf.nn.dropout(out_conv4, keep_prob)
	#hidden_conv4 = tf.nn.max_pool3d(dropout_conv4, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv3 ,the output size is batch_sizex4x4x4x128([batch_size,in_deep,width,height,output_deep])
	# all feature map flatten to one dimension vector,this vector will be much long
	flattened_conv4 = tf.reshape(dropout_conv4,[-1,512])
	w_fc1 = tf.Variable(tf.random_normal([512,512],stddev=0.1),name='w_fc1')
	b_fc1 = tf.Variable(tf.random_normal(shape=[512], stddev=0.1), name='b_fc1')
	out_fc1 = tf.nn.relu(tf.add(tf.matmul(flattened_conv4,w_fc1),b_fc1))
	dropout_fc1 = tf.nn.dropout(out_fc1,keep_prob)

	w_fc2 = tf.Variable(tf.random_normal([512, 2], stddev=0.1), name='w_fc2')
	b_fc2 = tf.Variable(tf.random_normal(shape=[2], stddev=0.1), name='b_fc2')
	out_fc2 = tf.add(tf.matmul(out_fc1, w_fc2), b_fc2)

	out_sm = tf.nn.softmax(out_fc2)

	#return hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, dropout_fc1, w_fc2, b_fc2, out_fc2, out_sm
	return out_fc2, out_sm

def volume_bnnet_l6_56(input):
	bn_variance_epsilon = tf.constant(0.0000000000001, dtype=tf.float32)

	w_conv1 = tf.Variable(tf.random_normal([3, 3, 3, 1, 16], stddev=0.1), dtype=tf.float32, name='w_conv1')
	out_conv1 = tf.nn.conv3d(input, w_conv1, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean1, bn_var1 = tf.nn.moments(out_conv1, [i for i in range(len(out_conv1.shape))])
	r_bn1 = tf.Variable([1], dtype=tf.float32, name='r_bn1')
	b_bn1 = tf.Variable(tf.random_normal([16], stddev=0.1), dtype=tf.float32, name='b_bn1')
	out_bn1 = tf.nn.batch_normalization(out_conv1, bn_mean1, bn_var1, b_bn1, r_bn1, bn_variance_epsilon)
	hidden_conv1 = tf.nn.max_pool3d(tf.nn.relu(out_bn1), strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1],
					padding='SAME')

	# after conv1 ,the output size is batch_sizex27x27x27x16([batch_size,in_deep,width,height,output_deep])
	w_conv2 = tf.Variable(tf.random_normal([4, 4, 4, 16, 32], stddev=0.1), dtype=tf.float32, name='w_conv2')
	out_conv2 = tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean2, bn_var2 = tf.nn.moments(out_conv2, [i for i in range(len(out_conv2.shape))])
	r_bn2 = tf.Variable([1], dtype=tf.float32, name='r_bn2')
	b_bn2 = tf.Variable(tf.random_normal([32], stddev=0.1), dtype=tf.float32, name='b_bn2')
	out_bn2 = tf.nn.batch_normalization(out_conv2, bn_mean2, bn_var2, b_bn2, r_bn2, bn_variance_epsilon)
	hidden_conv2 = tf.nn.max_pool3d(tf.nn.relu(out_bn2), strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1],
					padding='SAME')

	# after conv2 ,the output size is batch_sizex12x12x12x32([batch_size,in_deep,width,height,output_deep])
	w_conv3 = tf.Variable(tf.random_normal([5, 5, 5, 32, 64], stddev=0.1), dtype=tf.float32, name='w_conv3')
	out_conv3 = tf.nn.conv3d(hidden_conv2, w_conv3, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean3, bn_var3 = tf.nn.moments(out_conv3, [i for i in range(len(out_conv3.shape))])
	r_bn3 = tf.Variable([1], dtype=tf.float32, name='r_bn3')
	b_bn3 = tf.Variable(tf.random_normal([64], stddev=0.1), dtype=tf.float32, name='b_bn3')
	out_bn3 = tf.nn.batch_normalization(out_conv3, bn_mean3, bn_var3, b_bn3, r_bn3, bn_variance_epsilon)
	hidden_conv3 = tf.nn.max_pool3d(tf.nn.relu(out_bn3), strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1],
					padding='SAME')

	# after conv3 ,the output size is batch_sizex4x4x4x64([batch_size,in_deep,width,height,output_deep])
	w_conv4 = tf.Variable(tf.random_normal([4, 4, 4, 64, 128], stddev=0.1), dtype=tf.float32, name='w_conv4')
	out_conv4 = tf.nn.conv3d(hidden_conv3, w_conv4, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean4, bn_var4 = tf.nn.moments(out_conv4, [i for i in range(len(out_conv4.shape))])
	r_bn4 = tf.Variable([1], dtype=tf.float32, name='r_bn4')
	b_bn4 = tf.Variable(tf.random_normal([128], stddev=0.1), dtype=tf.float32, name='b_bn4')
	out_bn4 = tf.nn.batch_normalization(out_conv4, bn_mean4, bn_var4, b_bn4, r_bn4, bn_variance_epsilon)
	hidden_conv4 = tf.nn.relu(out_bn4)

	# after conv3 ,the output size is batch_sizex4x4x4x128([batch_size,in_deep,width,height,output_deep])
	# all feature map flatten to one dimension vector,this vector will be much long
	flattened_conv4 = tf.reshape(hidden_conv4, [-1, 128])
	w_fc1 = tf.Variable(tf.random_normal([128, 128], stddev=0.1), name='w_fc1')
	out_fc1 = tf.matmul(flattened_conv4, w_fc1)
	r_bn5 = tf.Variable([1], dtype=tf.float32, name='r_bn5')
	b_bn5 = tf.Variable(tf.random_normal([128], stddev=0.1), dtype=tf.float32, name='b_bn5')
	bn_mean5, bn_var5 = tf.nn.moments(out_fc1, [i for i in range(len(out_fc1.shape))])
	out_bn5 = tf.nn.batch_normalization(out_fc1, bn_mean5, bn_var5, b_bn5, r_bn5, bn_variance_epsilon)
	out_rl5 = tf.nn.relu(out_bn5)
	# dropout_fc1 = tf.nn.dropout(out_fc1,keep_prob)

	w_fc2 = tf.Variable(tf.random_normal([128, 2], stddev=0.1), name='w_fc2')
	b_fc2 = tf.Variable(tf.random_normal(shape=[2], stddev=0.1), name='b_fc2')
	out_fc2 = tf.add(tf.matmul(out_rl5, w_fc2), b_fc2)

	out_sm = tf.nn.softmax(out_fc2)

	# return b_bn1, w_conv1, w_conv2, out_conv1, out_bn1, hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, out_fc2, out_sm
	return out_fc2, out_sm


def volume_bnnet2_l6_56(input):
	bn_variance_epsilon = tf.constant(0.0000000000001, dtype=tf.float32)

	w_conv1 = tf.Variable(tf.random_normal([3, 3, 3, 1, 64], stddev=0.1), dtype=tf.float32, name='w_conv1')
	out_conv1 = tf.nn.conv3d(input, w_conv1, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean1, bn_var1 = tf.nn.moments(out_conv1, [i for i in range(len(out_conv1.shape))])
	r_bn1 = tf.Variable([1], dtype=tf.float32, name='r_bn1')
	b_bn1 = tf.Variable(tf.random_normal([64], stddev=0.1), dtype=tf.float32, name='b_bn1')
	out_bn1 = tf.nn.batch_normalization(out_conv1, bn_mean1, bn_var1, b_bn1, r_bn1, bn_variance_epsilon)
	hidden_conv1 = tf.nn.max_pool3d(tf.nn.relu(out_bn1), strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv1 ,the output size is batch_sizex27x27x27x16([batch_size,in_deep,width,height,output_deep])
	w_conv2 = tf.Variable(tf.random_normal([4, 4, 4, 64, 128], stddev=0.1), dtype=tf.float32, name='w_conv2')
	out_conv2 = tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean2, bn_var2 = tf.nn.moments(out_conv2, [i for i in range(len(out_conv2.shape))])
	r_bn2 = tf.Variable([1], dtype=tf.float32, name='r_bn2')
	b_bn2 = tf.Variable(tf.random_normal([128], stddev=0.1), dtype=tf.float32, name='b_bn2')
	out_bn2 = tf.nn.batch_normalization(out_conv2, bn_mean2, bn_var2, b_bn2, r_bn2, bn_variance_epsilon)
	hidden_conv2 = tf.nn.max_pool3d(tf.nn.relu(out_bn2), strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv2 ,the output size is batch_sizex12x12x12x32([batch_size,in_deep,width,height,output_deep])
	w_conv3 = tf.Variable(tf.random_normal([5, 5, 5, 128, 256], stddev=0.1), dtype=tf.float32, name='w_conv3')
	out_conv3 = tf.nn.conv3d(hidden_conv2, w_conv3, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean3, bn_var3 = tf.nn.moments(out_conv3, [i for i in range(len(out_conv3.shape))])
	r_bn3 = tf.Variable([1], dtype=tf.float32, name='r_bn3')
	b_bn3 = tf.Variable(tf.random_normal([256], stddev=0.1), dtype=tf.float32, name='b_bn3')
	out_bn3 = tf.nn.batch_normalization(out_conv3, bn_mean3, bn_var3, b_bn3, r_bn3, bn_variance_epsilon)
	hidden_conv3 = tf.nn.max_pool3d(tf.nn.relu(out_bn3), strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv3 ,the output size is batch_sizex4x4x4x64([batch_size,in_deep,width,height,output_deep])
	w_conv4 = tf.Variable(tf.random_normal([4, 4, 4, 256, 512], stddev=0.1), dtype=tf.float32, name='w_conv4')
	out_conv4 = tf.nn.conv3d(hidden_conv3, w_conv4, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean4, bn_var4 = tf.nn.moments(out_conv4, [i for i in range(len(out_conv4.shape))])
	r_bn4 = tf.Variable([1], dtype=tf.float32, name='r_bn4')
	b_bn4 = tf.Variable(tf.random_normal([512], stddev=0.1), dtype=tf.float32, name='b_bn4')
	out_bn4 = tf.nn.batch_normalization(out_conv4, bn_mean4, bn_var4, b_bn4, r_bn4, bn_variance_epsilon)
	hidden_conv4 = tf.nn.relu(out_bn4)

	# after conv3 ,the output size is batch_sizex4x4x4x128([batch_size,in_deep,width,height,output_deep])
	# all feature map flatten to one dimension vector,this vector will be much long
	flattened_conv4 = tf.reshape(hidden_conv4, [-1, 512])
	w_fc1 = tf.Variable(tf.random_normal([512, 512], stddev=0.1), name='w_fc1')
	out_fc1 = tf.matmul(flattened_conv4, w_fc1)
	r_bn5 = tf.Variable([1], dtype=tf.float32, name='r_bn5')
	b_bn5 = tf.Variable(tf.random_normal([512], stddev=0.1), dtype=tf.float32, name='b_bn5')
	bn_mean5, bn_var5 = tf.nn.moments(out_fc1, [i for i in range(len(out_fc1.shape))])
	out_bn5 = tf.nn.batch_normalization(out_fc1, bn_mean5, bn_var5, b_bn5, r_bn5, bn_variance_epsilon)
	out_rl5 = tf.nn.relu(out_bn5)
	# dropout_fc1 = tf.nn.dropout(out_fc1,keep_prob)

	w_fc2 = tf.Variable(tf.random_normal([512, 2], stddev=0.1), name='w_fc2')
	b_fc2 = tf.Variable(tf.random_normal(shape=[2], stddev=0.1), name='b_fc2')
	out_fc2 = tf.add(tf.matmul(out_rl5, w_fc2), b_fc2)

	out_sm = tf.nn.softmax(out_fc2)

	#return b_bn1, w_conv1, w_conv2, out_conv1, out_bn1, hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, out_fc2, out_sm
	return out_fc2, out_sm

def volume_bnnet3_l6_56(input):
	bn_variance_epsilon = tf.constant(0.0000000000001, dtype=tf.float32)

	w_conv1 = tf.Variable(tf.random_normal([3, 3, 3, 1, 64], stddev=0.1), dtype=tf.float32, name='w_conv1')
	out_conv1 = tf.nn.conv3d(input, w_conv1, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean1, bn_var1 = tf.nn.moments(out_conv1, [i for i in range(len(out_conv1.shape))])
	#r_bn1 = tf.Variable([1], dtype=tf.float32, name='r_bn1')
	b_bn1 = tf.Variable(tf.random_normal([64], stddev=0.1), dtype=tf.float32, name='b_bn1')
	out_bn1 = tf.nn.batch_normalization(out_conv1, bn_mean1, bn_var1, b_bn1, None, bn_variance_epsilon)
	hidden_conv1 = tf.nn.max_pool3d(tf.nn.relu(out_bn1), strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv1 ,the output size is batch_sizex27x27x27x16([batch_size,in_deep,width,height,output_deep])
	w_conv2 = tf.Variable(tf.random_normal([4, 4, 4, 64, 128], stddev=0.1), dtype=tf.float32, name='w_conv2')
	out_conv2 = tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean2, bn_var2 = tf.nn.moments(out_conv2, [i for i in range(len(out_conv2.shape))])
	#r_bn2 = tf.Variable([1], dtype=tf.float32, name='r_bn2')
	b_bn2 = tf.Variable(tf.random_normal([128], stddev=0.1), dtype=tf.float32, name='b_bn2')
	out_bn2 = tf.nn.batch_normalization(out_conv2, bn_mean2, bn_var2, b_bn2, None, bn_variance_epsilon)
	hidden_conv2 = tf.nn.max_pool3d(tf.nn.relu(out_bn2), strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv2 ,the output size is batch_sizex12x12x12x32([batch_size,in_deep,width,height,output_deep])
	w_conv3 = tf.Variable(tf.random_normal([5, 5, 5, 128, 256], stddev=0.1), dtype=tf.float32, name='w_conv3')
	out_conv3 = tf.nn.conv3d(hidden_conv2, w_conv3, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean3, bn_var3 = tf.nn.moments(out_conv3, [i for i in range(len(out_conv3.shape))])
	#r_bn3 = tf.Variable([1], dtype=tf.float32, name='r_bn3')
	b_bn3 = tf.Variable(tf.random_normal([256], stddev=0.1), dtype=tf.float32, name='b_bn3')
	out_bn3 = tf.nn.batch_normalization(out_conv3, bn_mean3, bn_var3, b_bn3, None, bn_variance_epsilon)
	hidden_conv3 = tf.nn.max_pool3d(tf.nn.relu(out_bn3), strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv3 ,the output size is batch_sizex4x4x4x64([batch_size,in_deep,width,height,output_deep])
	w_conv4 = tf.Variable(tf.random_normal([4, 4, 4, 256, 512], stddev=0.1), dtype=tf.float32, name='w_conv4')
	out_conv4 = tf.nn.conv3d(hidden_conv3, w_conv4, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean4, bn_var4 = tf.nn.moments(out_conv4, [i for i in range(len(out_conv4.shape))])
	#r_bn4 = tf.Variable([1], dtype=tf.float32, name='r_bn4')
	b_bn4 = tf.Variable(tf.random_normal([512], stddev=0.1), dtype=tf.float32, name='b_bn4')
	out_bn4 = tf.nn.batch_normalization(out_conv4, bn_mean4, bn_var4, b_bn4, None, bn_variance_epsilon)
	hidden_conv4 = tf.nn.relu(out_bn4)

	# after conv3 ,the output size is batch_sizex4x4x4x128([batch_size,in_deep,width,height,output_deep])
	# all feature map flatten to one dimension vector,this vector will be much long
	flattened_conv4 = tf.reshape(hidden_conv4, [-1, 512])
	w_fc1 = tf.Variable(tf.random_normal([512, 512], stddev=0.1), name='w_fc1')
	out_fc1 = tf.matmul(flattened_conv4, w_fc1)
	#r_bn5 = tf.Variable([1], dtype=tf.float32, name='r_bn5')
	b_bn5 = tf.Variable(tf.random_normal([512], stddev=0.1), dtype=tf.float32, name='b_bn5')
	bn_mean5, bn_var5 = tf.nn.moments(out_fc1, [i for i in range(len(out_fc1.shape))])
	out_bn5 = tf.nn.batch_normalization(out_fc1, bn_mean5, bn_var5, b_bn5, None, bn_variance_epsilon)
	out_rl5 = tf.nn.relu(out_bn5)
	# dropout_fc1 = tf.nn.dropout(out_fc1,keep_prob)

	w_fc2 = tf.Variable(tf.random_normal([512, 2], stddev=0.1), name='w_fc2')
	b_fc2 = tf.Variable(tf.random_normal(shape=[2], stddev=0.1), name='b_fc2')
	out_fc2 = tf.add(tf.matmul(out_rl5, w_fc2), b_fc2)

	out_sm = tf.nn.softmax(out_fc2)

	#return b_bn1, w_conv1, w_conv2, out_conv1, out_bn1, hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, out_fc2, out_sm
	return out_fc2, out_sm

def volume_bnnet_zerobias_l6_56(input):
	bn_variance_epsilon = tf.constant(0.0000000000001, dtype=tf.float32)

	w_conv1 = tf.Variable(tf.random_normal([3, 3, 3, 1, 64], stddev=0.1), dtype=tf.float32, name='w_conv1')
	out_conv1 = tf.nn.conv3d(input, w_conv1, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean1, bn_var1 = tf.nn.moments(out_conv1, [i for i in range(len(out_conv1.shape))])
	r_bn1 = tf.Variable([1], dtype=tf.float32, name='r_bn1')
	b_bn1 = tf.Variable(tf.zeros([64]), dtype=tf.float32, name='b_bn1')
	out_bn1 = tf.nn.batch_normalization(out_conv1, bn_mean1, bn_var1, b_bn1, r_bn1, bn_variance_epsilon)
	hidden_conv1 = tf.nn.max_pool3d(tf.nn.relu(out_bn1), strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv1 ,the output size is batch_sizex27x27x27x16([batch_size,in_deep,width,height,output_deep])
	w_conv2 = tf.Variable(tf.random_normal([4, 4, 4, 64, 128], stddev=0.1), dtype=tf.float32, name='w_conv2')
	out_conv2 = tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean2, bn_var2 = tf.nn.moments(out_conv2, [i for i in range(len(out_conv2.shape))])
	r_bn2 = tf.Variable([1], dtype=tf.float32, name='r_bn2')
	b_bn2 = tf.Variable(tf.zeros([128]), dtype=tf.float32, name='b_bn2')
	out_bn2 = tf.nn.batch_normalization(out_conv2, bn_mean2, bn_var2, b_bn2, r_bn2, bn_variance_epsilon)
	hidden_conv2 = tf.nn.max_pool3d(tf.nn.relu(out_bn2), strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv2 ,the output size is batch_sizex12x12x12x32([batch_size,in_deep,width,height,output_deep])
	w_conv3 = tf.Variable(tf.random_normal([5, 5, 5, 128, 256], stddev=0.1), dtype=tf.float32, name='w_conv3')
	out_conv3 = tf.nn.conv3d(hidden_conv2, w_conv3, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean3, bn_var3 = tf.nn.moments(out_conv3, [i for i in range(len(out_conv3.shape))])
	r_bn3 = tf.Variable([1], dtype=tf.float32, name='r_bn3')
	b_bn3 = tf.Variable(tf.zeros([256]), dtype=tf.float32, name='b_bn3')
	out_bn3 = tf.nn.batch_normalization(out_conv3, bn_mean3, bn_var3, b_bn3, r_bn3, bn_variance_epsilon)
	hidden_conv3 = tf.nn.max_pool3d(tf.nn.relu(out_bn3), strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv3 ,the output size is batch_sizex4x4x4x64([batch_size,in_deep,width,height,output_deep])
	w_conv4 = tf.Variable(tf.random_normal([4, 4, 4, 256, 512], stddev=0.1), dtype=tf.float32, name='w_conv4')
	out_conv4 = tf.nn.conv3d(hidden_conv3, w_conv4, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean4, bn_var4 = tf.nn.moments(out_conv4, [i for i in range(len(out_conv4.shape))])
	r_bn4 = tf.Variable([1], dtype=tf.float32, name='r_bn4')
	b_bn4 = tf.Variable(tf.zeros([512]), dtype=tf.float32, name='b_bn4')
	out_bn4 = tf.nn.batch_normalization(out_conv4, bn_mean4, bn_var4, b_bn4, r_bn4, bn_variance_epsilon)
	hidden_conv4 = tf.nn.relu(out_bn4)

	# after conv3 ,the output size is batch_sizex4x4x4x128([batch_size,in_deep,width,height,output_deep])
	# all feature map flatten to one dimension vector,this vector will be much long
	flattened_conv4 = tf.reshape(hidden_conv4, [-1, 512])
	w_fc1 = tf.Variable(tf.random_normal([512, 512], stddev=0.1), name='w_fc1')
	out_fc1 = tf.matmul(flattened_conv4, w_fc1)
	r_bn5 = tf.Variable([1], dtype=tf.float32, name='r_bn5')
	b_bn5 = tf.Variable(tf.zeros([512]), dtype=tf.float32, name='b_bn5')
	bn_mean5, bn_var5 = tf.nn.moments(out_fc1, [i for i in range(len(out_fc1.shape))])
	out_bn5 = tf.nn.batch_normalization(out_fc1, bn_mean5, bn_var5, b_bn5, r_bn5, bn_variance_epsilon)
	out_rl5 = tf.nn.relu(out_bn5)
	# dropout_fc1 = tf.nn.dropout(out_fc1,keep_prob)

	w_fc2 = tf.Variable(tf.random_normal([512, 2], stddev=0.1), name='w_fc2')
	b_fc2 = tf.Variable(tf.zeros([2]), name='b_fc2')
	out_fc2 = tf.add(tf.matmul(out_rl5, w_fc2), b_fc2)

	out_sm = tf.nn.softmax(out_fc2)

	#return b_bn1, w_conv1, w_conv2, out_conv1, out_bn1, hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, out_fc2, out_sm
	return out_fc2, out_sm
	
def volume_bnnet_flbias_l6_56(input, positive_confidence=0.01):
	#batch normalization net for focal loss training with bias fine tuned for positive_confidence
	bn_variance_epsilon = tf.constant(0.0000000000001, dtype=tf.float32)

	w_conv1 = tf.Variable(tf.random_normal([3, 3, 3, 1, 64], stddev=0.1), dtype=tf.float32, name='w_conv1')
	out_conv1 = tf.nn.conv3d(input, w_conv1, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean1, bn_var1 = tf.nn.moments(out_conv1, [i for i in range(len(out_conv1.shape))])
	r_bn1 = tf.Variable([1], dtype=tf.float32, name='r_bn1')
	b_bn1 = tf.Variable(tf.random_normal([64], stddev=0.1), dtype=tf.float32, name='b_bn1')
	out_bn1 = tf.nn.batch_normalization(out_conv1, bn_mean1, bn_var1, b_bn1, r_bn1, bn_variance_epsilon)
	hidden_conv1 = tf.nn.max_pool3d(tf.nn.relu(out_bn1), strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv1 ,the output size is batch_sizex27x27x27x16([batch_size,in_deep,width,height,output_deep])
	w_conv2 = tf.Variable(tf.random_normal([4, 4, 4, 64, 128], stddev=0.1), dtype=tf.float32, name='w_conv2')
	out_conv2 = tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean2, bn_var2 = tf.nn.moments(out_conv2, [i for i in range(len(out_conv2.shape))])
	r_bn2 = tf.Variable([1], dtype=tf.float32, name='r_bn2')
	b_bn2 = tf.Variable(tf.random_normal([128], stddev=0.1), dtype=tf.float32, name='b_bn2')
	out_bn2 = tf.nn.batch_normalization(out_conv2, bn_mean2, bn_var2, b_bn2, r_bn2, bn_variance_epsilon)
	hidden_conv2 = tf.nn.max_pool3d(tf.nn.relu(out_bn2), strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv2 ,the output size is batch_sizex12x12x12x32([batch_size,in_deep,width,height,output_deep])
	w_conv3 = tf.Variable(tf.random_normal([5, 5, 5, 128, 256], stddev=0.1), dtype=tf.float32, name='w_conv3')
	out_conv3 = tf.nn.conv3d(hidden_conv2, w_conv3, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean3, bn_var3 = tf.nn.moments(out_conv3, [i for i in range(len(out_conv3.shape))])
	r_bn3 = tf.Variable([1], dtype=tf.float32, name='r_bn3')
	b_bn3 = tf.Variable(tf.random_normal([256], stddev=0.1), dtype=tf.float32, name='b_bn3')
	out_bn3 = tf.nn.batch_normalization(out_conv3, bn_mean3, bn_var3, b_bn3, r_bn3, bn_variance_epsilon)
	hidden_conv3 = tf.nn.max_pool3d(tf.nn.relu(out_bn3), strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv3 ,the output size is batch_sizex4x4x4x64([batch_size,in_deep,width,height,output_deep])
	w_conv4 = tf.Variable(tf.random_normal([4, 4, 4, 256, 512], stddev=0.1), dtype=tf.float32, name='w_conv4')
	out_conv4 = tf.nn.conv3d(hidden_conv3, w_conv4, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean4, bn_var4 = tf.nn.moments(out_conv4, [i for i in range(len(out_conv4.shape))])
	r_bn4 = tf.Variable([1], dtype=tf.float32, name='r_bn4')
	b_bn4 = tf.Variable(tf.random_normal([512], stddev=0.1), dtype=tf.float32, name='b_bn4')
	out_bn4 = tf.nn.batch_normalization(out_conv4, bn_mean4, bn_var4, b_bn4, r_bn4, bn_variance_epsilon)
	hidden_conv4 = tf.nn.relu(out_bn4)

	# after conv3 ,the output size is batch_sizex4x4x4x128([batch_size,in_deep,width,height,output_deep])
	# all feature map flatten to one dimension vector,this vector will be much long
	flattened_conv4 = tf.reshape(hidden_conv4, [-1, 512])
	w_fc1 = tf.Variable(tf.random_normal([512, 512], stddev=0.1), name='w_fc1')
	out_fc1 = tf.matmul(flattened_conv4, w_fc1)
	r_bn5 = tf.Variable([1], dtype=tf.float32, name='r_bn5')
	b_bn5 = tf.Variable(tf.random_normal([512], stddev=0.1), dtype=tf.float32, name='b_bn5')
	bn_mean5, bn_var5 = tf.nn.moments(out_fc1, [i for i in range(len(out_fc1.shape))])
	out_bn5 = tf.nn.batch_normalization(out_fc1, bn_mean5, bn_var5, b_bn5, r_bn5, bn_variance_epsilon)
	out_rl5 = tf.nn.relu(out_bn5)
	# dropout_fc1 = tf.nn.dropout(out_fc1,keep_prob)

	w_fc2 = tf.Variable(tf.random_normal([512, 2], stddev=0.1), name='w_fc2')
	#b_fc2 = tf.Variable(tf.random_normal(shape=[2], mean=0, stddev=0.1) + tf.constant([-math.log((1-positive_confidence)/positive_confidence), math.log((1-positive_confidence)/positive_confidence)]), name='b_fc2')
	b_fc2 = tf.Variable([-math.log((1-positive_confidence)/positive_confidence), math.log((1-positive_confidence)/positive_confidence)], name='b_fc2')
	out_fc2 = tf.add(tf.matmul(out_rl5, w_fc2), b_fc2)

	out_sm = tf.nn.softmax(out_fc2)

	#return b_bn1, w_conv1, w_conv2, out_conv1, out_bn1, hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, out_fc2, out_sm
	return out_fc2, out_sm
	
def volume_bndo_flbias_l6_56(input, positive_confidence=0.01, dropout_rate=0.3):
	#batch normalization net for focal loss training with bias fine tuned for positive_confidence
	#dropout layer adapted
	bn_variance_epsilon = tf.constant(0.0000000000001, dtype=tf.float32)
	keep_prob = tf.constant(1-dropout_rate, dtype=tf.float32)

	w_conv1 = tf.Variable(tf.random_normal([3, 3, 3, 1, 64], stddev=0.1), dtype=tf.float32, name='w_conv1')
	out_conv1 = tf.nn.conv3d(input, w_conv1, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean1, bn_var1 = tf.nn.moments(out_conv1, [i for i in range(len(out_conv1.shape))])
	r_bn1 = tf.Variable([1], dtype=tf.float32, name='r_bn1')
	b_bn1 = tf.Variable(tf.random_normal([64], stddev=0.1), dtype=tf.float32, name='b_bn1')
	out_bn1 = tf.nn.batch_normalization(out_conv1, bn_mean1, bn_var1, b_bn1, r_bn1, bn_variance_epsilon)
	out_dropout1 = tf.nn.dropout(tf.nn.relu(out_bn1), keep_prob)
	hidden_conv1 = tf.nn.max_pool3d(out_dropout1, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')
	#del out_conv1, out_bn1, out_dropout1

	# after conv1 ,the output size is batch_sizex27x27x27x16([batch_size,in_deep,width,height,output_deep])
	w_conv2 = tf.Variable(tf.random_normal([4, 4, 4, 64, 128], stddev=0.1), dtype=tf.float32, name='w_conv2')
	out_conv2 = tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean2, bn_var2 = tf.nn.moments(out_conv2, [i for i in range(len(out_conv2.shape))])
	r_bn2 = tf.Variable([1], dtype=tf.float32, name='r_bn2')
	b_bn2 = tf.Variable(tf.random_normal([128], stddev=0.1), dtype=tf.float32, name='b_bn2')
	out_bn2 = tf.nn.batch_normalization(out_conv2, bn_mean2, bn_var2, b_bn2, r_bn2, bn_variance_epsilon)
	out_dropout2 = tf.nn.dropout(tf.nn.relu(out_bn2), keep_prob)
	hidden_conv2 = tf.nn.max_pool3d(out_dropout2, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')
	#del out_conv2, out_bn2, out_dropout2

	# after conv2 ,the output size is batch_sizex12x12x12x32([batch_size,in_deep,width,height,output_deep])
	w_conv3 = tf.Variable(tf.random_normal([5, 5, 5, 128, 256], stddev=0.1), dtype=tf.float32, name='w_conv3')
	out_conv3 = tf.nn.conv3d(hidden_conv2, w_conv3, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean3, bn_var3 = tf.nn.moments(out_conv3, [i for i in range(len(out_conv3.shape))])
	r_bn3 = tf.Variable([1], dtype=tf.float32, name='r_bn3')
	b_bn3 = tf.Variable(tf.random_normal([256], stddev=0.1), dtype=tf.float32, name='b_bn3')
	out_bn3 = tf.nn.batch_normalization(out_conv3, bn_mean3, bn_var3, b_bn3, r_bn3, bn_variance_epsilon)
	out_dropout3 = tf.nn.dropout(tf.nn.relu(out_bn3), keep_prob)
	hidden_conv3 = tf.nn.max_pool3d(out_dropout3, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')
	#del out_conv3, out_bn3, out_dropout3

	# after conv3 ,the output size is batch_sizex4x4x4x64([batch_size,in_deep,width,height,output_deep])
	w_conv4 = tf.Variable(tf.random_normal([4, 4, 4, 256, 512], stddev=0.1), dtype=tf.float32, name='w_conv4')
	out_conv4 = tf.nn.conv3d(hidden_conv3, w_conv4, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean4, bn_var4 = tf.nn.moments(out_conv4, [i for i in range(len(out_conv4.shape))])
	r_bn4 = tf.Variable([1], dtype=tf.float32, name='r_bn4')
	b_bn4 = tf.Variable(tf.random_normal([512], stddev=0.1), dtype=tf.float32, name='b_bn4')
	out_bn4 = tf.nn.batch_normalization(out_conv4, bn_mean4, bn_var4, b_bn4, r_bn4, bn_variance_epsilon)
	out_dropout4 = tf.nn.dropout(tf.nn.relu(out_bn4), keep_prob)
	hidden_conv4 = tf.nn.max_pool3d(out_dropout4, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')
	#del out_conv4, out_bn4, out_dropout4

	# after conv3 ,the output size is batch_sizex4x4x4x128([batch_size,in_deep,width,height,output_deep])
	# all feature map flatten to one dimension vector,this vector will be much long
	flattened_conv4 = tf.reshape(hidden_conv4, [-1, 512])
	w_fc1 = tf.Variable(tf.random_normal([512, 512], stddev=0.1), name='w_fc1')
	out_fc1 = tf.matmul(flattened_conv4, w_fc1)
	r_bn5 = tf.Variable([1], dtype=tf.float32, name='r_bn5')
	b_bn5 = tf.Variable(tf.random_normal([512], stddev=0.1), dtype=tf.float32, name='b_bn5')
	bn_mean5, bn_var5 = tf.nn.moments(out_fc1, [i for i in range(len(out_fc1.shape))])
	out_bn5 = tf.nn.batch_normalization(out_fc1, bn_mean5, bn_var5, b_bn5, r_bn5, bn_variance_epsilon)
	out_dropout5 = tf.nn.dropout(tf.nn.relu(out_bn5), keep_prob)
	#del out_fc1, out_bn5

	w_fc2 = tf.Variable(tf.random_normal([512, 2], stddev=0.1), name='w_fc2')
	#b_fc2 = tf.Variable(tf.random_normal(shape=[2], mean=0, stddev=0.1) + tf.constant([-math.log((1-positive_confidence)/positive_confidence), math.log((1-positive_confidence)/positive_confidence)]), name='b_fc2')
	b_fc2 = tf.Variable([-math.log((1-positive_confidence)/positive_confidence), math.log((1-positive_confidence)/positive_confidence)], name='b_fc2')
	out_fc2 = tf.add(tf.matmul(out_dropout5, w_fc2), b_fc2)

	out_sm = tf.nn.softmax(out_fc2)

	#return b_bn1, w_conv1, w_conv2, out_conv1, out_bn1, hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, out_fc2, out_sm
	return out_fc2, out_sm
	
def volume_bndo_flbias_l5_42(input, positive_confidence=0.5, dropout_rate=0.3):
	#batch normalization net for focal loss training with bias fine tuned for positive_confidence
	#dropout layer adapted
	bn_variance_epsilon = tf.constant(0.0000000000001, dtype=tf.float32)
	keep_prob = tf.constant(1-dropout_rate, dtype=tf.float32)

	w_conv1 = tf.Variable(tf.random_normal([5, 5, 5, 1, 64], stddev=0.1), dtype=tf.float32, name='w_conv1')
	out_conv1 = tf.nn.conv3d(input, w_conv1, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean1, bn_var1 = tf.nn.moments(out_conv1, [i for i in range(len(out_conv1.shape))])
	r_bn1 = tf.Variable([1], dtype=tf.float32, name='r_bn1')
	b_bn1 = tf.Variable(tf.random_normal([64], stddev=0.1), dtype=tf.float32, name='b_bn1')
	out_bn1 = tf.nn.batch_normalization(out_conv1, bn_mean1, bn_var1, b_bn1, r_bn1, bn_variance_epsilon)
	out_dropout1 = tf.nn.dropout(tf.nn.relu(out_bn1), keep_prob)
	hidden_conv1 = tf.nn.max_pool3d(out_dropout1, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv1 ,the output volume size is 19x19x19([batch_size,in_deep,width,height,output_deep])
	w_conv2 = tf.Variable(tf.random_normal([4, 4, 4, 64, 128], stddev=0.1), dtype=tf.float32, name='w_conv2')
	out_conv2 = tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean2, bn_var2 = tf.nn.moments(out_conv2, [i for i in range(len(out_conv2.shape))])
	r_bn2 = tf.Variable([1], dtype=tf.float32, name='r_bn2')
	b_bn2 = tf.Variable(tf.random_normal([128], stddev=0.1), dtype=tf.float32, name='b_bn2')
	out_bn2 = tf.nn.batch_normalization(out_conv2, bn_mean2, bn_var2, b_bn2, r_bn2, bn_variance_epsilon)
	out_dropout2 = tf.nn.dropout(tf.nn.relu(out_bn2), keep_prob)
	hidden_conv2 = tf.nn.max_pool3d(out_dropout2, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv2 ,the output volume size is 8x8x8([batch_size,in_deep,width,height,output_deep])
	w_conv3 = tf.Variable(tf.random_normal([3, 3, 3, 128, 256], stddev=0.1), dtype=tf.float32, name='w_conv3')
	out_conv3 = tf.nn.conv3d(hidden_conv2, w_conv3, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean3, bn_var3 = tf.nn.moments(out_conv3, [i for i in range(len(out_conv3.shape))])
	r_bn3 = tf.Variable([1], dtype=tf.float32, name='r_bn3')
	b_bn3 = tf.Variable(tf.random_normal([256], stddev=0.1), dtype=tf.float32, name='b_bn3')
	out_bn3 = tf.nn.batch_normalization(out_conv3, bn_mean3, bn_var3, b_bn3, r_bn3, bn_variance_epsilon)
	out_dropout3 = tf.nn.dropout(tf.nn.relu(out_bn3), keep_prob)
	hidden_conv3 = tf.nn.max_pool3d(out_dropout3, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv3 ,the output volume size is 3x3x3([batch_size,in_deep,width,height,output_deep])
	w_conv4 = tf.Variable(tf.random_normal([3, 3, 3, 256, 1024], stddev=0.1), dtype=tf.float32, name='w_conv4')
	out_conv4 = tf.nn.conv3d(hidden_conv3, w_conv4, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean4, bn_var4 = tf.nn.moments(out_conv4, [i for i in range(len(out_conv4.shape))])
	r_bn4 = tf.Variable([1], dtype=tf.float32, name='r_bn4')
	b_bn4 = tf.Variable(tf.random_normal([1024], stddev=0.1), dtype=tf.float32, name='b_bn4')
	out_bn4 = tf.nn.batch_normalization(out_conv4, bn_mean4, bn_var4, b_bn4, r_bn4, bn_variance_epsilon)
	out_dropout4 = tf.nn.dropout(tf.nn.relu(out_bn4), keep_prob)

	# after conv3 ,the output size is batch_sizex1x1x1([batch_size,in_deep,width,height,output_deep])
	# all feature map flatten to one dimension vector,this vector will be much long
	flattened_conv4 = tf.reshape(out_dropout4, [-1, 1024])
	#w_fc1 = tf.Variable(tf.random_normal([512, 512], stddev=0.1), name='w_fc1')
	#out_fc1 = tf.matmul(flattened_conv4, w_fc1)
	#r_bn5 = tf.Variable([1], dtype=tf.float32, name='r_bn5')
	#b_bn5 = tf.Variable(tf.random_normal([512], stddev=0.1), dtype=tf.float32, name='b_bn5')
	#bn_mean5, bn_var5 = tf.nn.moments(out_fc1, [i for i in range(len(out_fc1.shape))])
	#out_bn5 = tf.nn.batch_normalization(out_fc1, bn_mean5, bn_var5, b_bn5, r_bn5, bn_variance_epsilon)
	#out_dropout5 = tf.nn.dropout(tf.nn.relu(out_bn5), keep_prob)

	w_fc1 = tf.Variable(tf.random_normal([1024, 2], stddev=0.1), name='w_fc1')
	#b_fc2 = tf.Variable(tf.random_normal(shape=[2], mean=0, stddev=0.1) + tf.constant([-math.log((1-positive_confidence)/positive_confidence), math.log((1-positive_confidence)/positive_confidence)]), name='b_fc2')
	b_fc1 = tf.Variable([-math.log((1-positive_confidence)/positive_confidence), math.log((1-positive_confidence)/positive_confidence)], name='b_fc1')
	out_fc1 = tf.add(tf.matmul(flattened_conv4, w_fc1), b_fc1)

	out_sm = tf.nn.softmax(out_fc1)

	#return b_bn1, w_conv1, w_conv2, out_conv1, out_bn1, hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, out_fc2, out_sm
	return out_fc1, out_sm
	
def volume_bndo_flbias_l6_40(input, training=True, positive_confidence=0.5, dropout_rate=0.3, batch_normalization_statistic=True, bn_params=None):
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
	w_conv3 = tf.Variable(tf.random_normal([2, 2, 2, 128, 256], stddev=0.1), dtype=tf.float32, name='w_conv3', trainable=training)
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
	hidden_conv3 = tf.nn.max_pool3d(out_dropout3, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv3 ,the output volume size is 3x3x3([batch_size,in_deep,width,height,output_deep])
	w_conv4 = tf.Variable(tf.random_normal([3, 3, 3, 256, 512], stddev=0.1), dtype=tf.float32, name='w_conv4', trainable=training)
	out_conv4 = tf.nn.conv3d(hidden_conv3, w_conv4, strides=[1, 1, 1, 1, 1], padding='VALID')
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
	
	outputs = {'conv1_out':hidden_conv1, 'conv2_out':hidden_conv2, 'conv3_out':hidden_conv3, 'conv4_out':out_dropout4, 'flattened_out':flattened_conv4, 'fc1_out':out_dropout5, 'last_out':out_fc2, 'sm_out':out_sm}
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
		
	#variables = [w_conv1, r_bn1, b_bn1, w_conv2, r_bn2, b_bn2, w_conv3, r_bn3, b_bn3, w_conv4, r_bn4, b_bn4, w_fc1, r_bn5, b_bn5, w_fc2, b_fc2]

	return outputs, variables, bn_pars
	
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
	
def volume_bndo_flbias_l5_30(input, training=True, positive_confidence=0.5, dropout_rate=0.3, batch_normalization_statistic=True, bn_params=None):
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

	# after conv1 ,the output volume size is 13x13x13([batch_size,in_deep,width,height,output_deep])
	w_conv2 = tf.Variable(tf.random_normal([2, 2, 2, 64, 128], stddev=0.1), dtype=tf.float32, name='w_conv2', trainable=training)
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

	# after conv2 ,the output volume size is 6x6x6([batch_size,in_deep,width,height,output_deep])
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
	hidden_conv3 = tf.nn.max_pool3d(out_dropout3, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv3 ,the output volume size is 2x2x2([batch_size,in_deep,width,height,output_deep])
	w_conv4 = tf.Variable(tf.random_normal([2, 2, 2, 256, 512], stddev=0.1), dtype=tf.float32, name='w_conv4', trainable=training)
	out_conv4 = tf.nn.conv3d(hidden_conv3, w_conv4, strides=[1, 1, 1, 1, 1], padding='VALID')
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
	#w_fc1 = tf.Variable(tf.random_normal([512, 512], stddev=0.1), name='w_fc1')
	#out_fc1 = tf.matmul(flattened_conv4, w_fc1)
	#r_bn5 = tf.Variable([1], dtype=tf.float32, name='r_bn5')
	#b_bn5 = tf.Variable(tf.random_normal([512], stddev=0.1), dtype=tf.float32, name='b_bn5')
	#bn_mean5, bn_var5 = tf.nn.moments(out_fc1, [i for i in range(len(out_fc1.shape))])
	#out_bn5 = tf.nn.batch_normalization(out_fc1, bn_mean5, bn_var5, b_bn5, r_bn5, bn_variance_epsilon)
	#out_dropout5 = tf.nn.dropout(tf.nn.relu(out_bn5), keep_prob)

	w_fc1 = tf.Variable(tf.random_normal([512, 2], stddev=0.1), name='w_fc1', trainable=training)
	#b_fc2 = tf.Variable(tf.random_normal(shape=[2], mean=0, stddev=0.1) + tf.constant([-math.log((1-positive_confidence)/positive_confidence), math.log((1-positive_confidence)/positive_confidence)]), name='b_fc2')
	b_fc1 = tf.Variable([-math.log((1-positive_confidence)/positive_confidence), math.log((1-positive_confidence)/positive_confidence)], name='b_fc1', trainable=training)
	out_fc1 = tf.add(tf.matmul(flattened_conv4, w_fc1), b_fc1)

	out_sm = tf.nn.softmax(out_fc1)
	
	outputs = {'conv1_out':hidden_conv1, 'conv2_out':hidden_conv2, 'conv3_out':hidden_conv3, 'conv4_out':out_dropout4, 'flattened_out':flattened_conv4, 'last_out':out_fc1, 'sm_out':out_sm}
	variables = {'w_conv1':w_conv1, 'r_bn1':r_bn1, 'b_bn1':b_bn1,
		     'w_conv2':w_conv2, 'r_bn2':r_bn2, 'b_bn2':b_bn2,
		     'w_conv3':w_conv3, 'r_bn3':r_bn3, 'b_bn3':b_bn3,
		     'w_conv4':w_conv4, 'r_bn4':r_bn4, 'b_bn4':b_bn4,
		     'w_fc1':w_fc1, 'b_fc1':b_fc1}
	
	if batch_normalization_statistic:
		bn_pars = []
		bn_pars.append([bn_mean1, bn_var1])
		bn_pars.append([bn_mean2, bn_var2])
		bn_pars.append([bn_mean3, bn_var3])
		bn_pars.append([bn_mean4, bn_var4])
	else:
		bn_pars = None

	return outputs, variables, bn_pars
	
def volume_bndo_flbias_l5_30_v2(input, positive_confidence=0.5, dropout_rate=0.3):
	#batch normalization net for focal loss training with bias fine tuned for positive_confidence
	#dropout layer adapted
	bn_variance_epsilon = tf.constant(0.0000000000001, dtype=tf.float32)
	keep_prob = tf.constant(1-dropout_rate, dtype=tf.float32)

	w_conv1 = tf.Variable(tf.random_normal([5, 5, 5, 1, 64], stddev=0.1), dtype=tf.float32, name='w_conv1')
	out_conv1 = tf.nn.conv3d(input, w_conv1, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean1, bn_var1 = tf.nn.moments(out_conv1, [i for i in range(len(out_conv1.shape))])
	r_bn1 = tf.Variable([1], dtype=tf.float32, name='r_bn1')
	b_bn1 = tf.Variable(tf.random_normal([64], stddev=0.1), dtype=tf.float32, name='b_bn1')
	out_bn1 = tf.nn.batch_normalization(out_conv1, bn_mean1, bn_var1, b_bn1, r_bn1, bn_variance_epsilon)
	out_dropout1 = tf.nn.dropout(tf.nn.relu(out_bn1), keep_prob)
	hidden_conv1 = tf.nn.max_pool3d(out_dropout1, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv1 ,the output volume size is 13x13x13([batch_size,in_deep,width,height,output_deep])
	w_conv2 = tf.Variable(tf.random_normal([4, 4, 4, 64, 128], stddev=0.1), dtype=tf.float32, name='w_conv2')
	out_conv2 = tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean2, bn_var2 = tf.nn.moments(out_conv2, [i for i in range(len(out_conv2.shape))])
	r_bn2 = tf.Variable([1], dtype=tf.float32, name='r_bn2')
	b_bn2 = tf.Variable(tf.random_normal([128], stddev=0.1), dtype=tf.float32, name='b_bn2')
	out_bn2 = tf.nn.batch_normalization(out_conv2, bn_mean2, bn_var2, b_bn2, r_bn2, bn_variance_epsilon)
	out_dropout2 = tf.nn.dropout(tf.nn.relu(out_bn2), keep_prob)
	hidden_conv2 = tf.nn.max_pool3d(out_dropout2, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')

	# after conv2 ,the output volume size is 5x5x5([batch_size,in_deep,width,height,output_deep])
	w_conv3 = tf.Variable(tf.random_normal([3, 3, 3, 128, 256], stddev=0.1), dtype=tf.float32, name='w_conv3')
	out_conv3 = tf.nn.conv3d(hidden_conv2, w_conv3, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean3, bn_var3 = tf.nn.moments(out_conv3, [i for i in range(len(out_conv3.shape))])
	r_bn3 = tf.Variable([1], dtype=tf.float32, name='r_bn3')
	b_bn3 = tf.Variable(tf.random_normal([256], stddev=0.1), dtype=tf.float32, name='b_bn3')
	out_bn3 = tf.nn.batch_normalization(out_conv3, bn_mean3, bn_var3, b_bn3, r_bn3, bn_variance_epsilon)
	out_dropout3 = tf.nn.dropout(tf.nn.relu(out_bn3), keep_prob)

	# after conv3 ,the output volume size is 3x3x3([batch_size,in_deep,width,height,output_deep])
	w_conv4 = tf.Variable(tf.random_normal([3, 3, 3, 256, 512], stddev=0.1), dtype=tf.float32, name='w_conv4')
	out_conv4 = tf.nn.conv3d(out_dropout3, w_conv4, strides=[1, 1, 1, 1, 1], padding='VALID')
	bn_mean4, bn_var4 = tf.nn.moments(out_conv4, [i for i in range(len(out_conv4.shape))])
	r_bn4 = tf.Variable([1], dtype=tf.float32, name='r_bn4')
	b_bn4 = tf.Variable(tf.random_normal([512], stddev=0.1), dtype=tf.float32, name='b_bn4')
	out_bn4 = tf.nn.batch_normalization(out_conv4, bn_mean4, bn_var4, b_bn4, r_bn4, bn_variance_epsilon)
	out_dropout4 = tf.nn.dropout(tf.nn.relu(out_bn4), keep_prob)

	# after conv3 ,the output size is batch_sizex1x1x1([batch_size,in_deep,width,height,output_deep])
	# all feature map flatten to one dimension vector,this vector will be much long
	flattened_conv4 = tf.reshape(out_dropout4, [-1, 512])
	#w_fc1 = tf.Variable(tf.random_normal([512, 512], stddev=0.1), name='w_fc1')
	#out_fc1 = tf.matmul(flattened_conv4, w_fc1)
	#r_bn5 = tf.Variable([1], dtype=tf.float32, name='r_bn5')
	#b_bn5 = tf.Variable(tf.random_normal([512], stddev=0.1), dtype=tf.float32, name='b_bn5')
	#bn_mean5, bn_var5 = tf.nn.moments(out_fc1, [i for i in range(len(out_fc1.shape))])
	#out_bn5 = tf.nn.batch_normalization(out_fc1, bn_mean5, bn_var5, b_bn5, r_bn5, bn_variance_epsilon)
	#out_dropout5 = tf.nn.dropout(tf.nn.relu(out_bn5), keep_prob)

	w_fc1 = tf.Variable(tf.random_normal([512, 2], stddev=0.1), name='w_fc1')
	#b_fc2 = tf.Variable(tf.random_normal(shape=[2], mean=0, stddev=0.1) + tf.constant([-math.log((1-positive_confidence)/positive_confidence), math.log((1-positive_confidence)/positive_confidence)]), name='b_fc2')
	b_fc1 = tf.Variable([-math.log((1-positive_confidence)/positive_confidence), math.log((1-positive_confidence)/positive_confidence)], name='b_fc1')
	out_fc1 = tf.add(tf.matmul(flattened_conv4, w_fc1), b_fc1)

	out_sm = tf.nn.softmax(out_fc1)

	#return b_bn1, w_conv1, w_conv2, out_conv1, out_bn1, hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, out_fc2, out_sm
	return out_fc1, out_sm

def volume_bndo_flbias_l5_20(input, training=True, positive_confidence=0.5, dropout_rate=0.3, batch_normalization_statistic=True, bn_params=None):
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

	# after conv1 ,the output volume size is 8x8x8([batch_size,in_deep,width,height,output_deep])
	w_conv2 = tf.Variable(tf.random_normal([3, 3, 3, 64, 128], stddev=0.1), dtype=tf.float32, name='w_conv2', trainable=training)
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

	# after conv2 ,the output volume size is 6x6x6([batch_size,in_deep,width,height,output_deep])
	w_conv3 = tf.Variable(tf.random_normal([4, 4, 4, 128, 256], stddev=0.1), dtype=tf.float32, name='w_conv3', trainable=training)
	out_conv3 = tf.nn.conv3d(out_dropout2, w_conv3, strides=[1, 1, 1, 1, 1], padding='VALID')
	if bn_params is None:
		bn_mean3, bn_var3 = tf.nn.moments(out_conv3, [i for i in range(len(out_conv3.shape))])
	else:
		bn_mean3 = tf.constant(bn_params[2][0], dtype=tf.float32, name='bn_mean3')
		bn_var3 = tf.constant(bn_params[2][1], dtype=tf.float32, name='bn_var3')
	r_bn3 = tf.Variable([1], dtype=tf.float32, name='r_bn3', trainable=training)
	b_bn3 = tf.Variable(tf.random_normal([256], stddev=0.1), dtype=tf.float32, name='b_bn3', trainable=training)
	out_bn3 = tf.nn.batch_normalization(out_conv3, bn_mean3, bn_var3, b_bn3, r_bn3, bn_variance_epsilon)
	out_dropout3 = tf.nn.dropout(tf.nn.relu(out_bn3), keep_prob)

	# after conv3 ,the output volume size is 3x3x3([batch_size,in_deep,width,height,output_deep])
	w_conv4 = tf.Variable(tf.random_normal([3, 3, 3, 256, 512], stddev=0.1), dtype=tf.float32, name='w_conv4', trainable=training)
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

	w_fc1 = tf.Variable(tf.random_normal([512, 2], stddev=0.1), name='w_fc1', trainable=training)
	#b_fc2 = tf.Variable(tf.random_normal(shape=[2], mean=0, stddev=0.1) + tf.constant([-math.log((1-positive_confidence)/positive_confidence), math.log((1-positive_confidence)/positive_confidence)]), name='b_fc2')
	b_fc1 = tf.Variable([-math.log((1-positive_confidence)/positive_confidence), math.log((1-positive_confidence)/positive_confidence)], name='b_fc1', trainable=training)
	out_fc1 = tf.add(tf.matmul(flattened_conv4, w_fc1), b_fc1)

	out_sm = tf.nn.softmax(out_fc1)
	
	outputs = {'conv1_out':hidden_conv1, 'conv2_out':out_dropout2, 'conv3_out':out_dropout3, 'conv4_out':out_dropout4, 'flattened_out':flattened_conv4, 'last_out':out_fc1, 'sm_out':out_sm}
	variables = {'w_conv1':w_conv1, 'r_bn1':r_bn1, 'b_bn1':b_bn1,
		     'w_conv2':w_conv2, 'r_bn2':r_bn2, 'b_bn2':b_bn2,
		     'w_conv3':w_conv3, 'r_bn3':r_bn3, 'b_bn3':b_bn3,
		     'w_conv4':w_conv4, 'r_bn4':r_bn4, 'b_bn4':b_bn4,
		     'w_fc1':w_fc1, 'b_fc1':b_fc1}
	
	if batch_normalization_statistic:
		bn_pars = []
		bn_pars.append([bn_mean1, bn_var1])
		bn_pars.append([bn_mean2, bn_var2])
		bn_pars.append([bn_mean3, bn_var3])
		bn_pars.append([bn_mean4, bn_var4])
	else:
		bn_pars = None

	return outputs, variables, bn_pars
	
def vote_fusion(predictions):
	predictions_clipped = [tf.reshape(predictions[0][:,0], [-1,1]), tf.reshape(predictions[1][:,0], [-1,1]), tf.reshape(predictions[2][:,0], [-1,1])]
	concpred = tf.keras.backend.concatenate(predictions_clipped, axis=1)
	return tf.reduce_max(concpred, axis=1)
	
def committe_fusion(predictions, weights=[0.3, 0.4, 0.3]):
	if len(predictions) != len(weights):
		print('length incorrect')
		return tf.add_n(predictions)/tf.constant(len(predictions), dtype=tf.float32)
	weighted_predictions = []
	for i in range(len(predictions)):
		weighted_predictions.append(predictions[i]*tf.constant(weights[i]))
	return tf.add_n(weighted_predictions)
	
def late_fusion(features, training=True, positive_confidence=0.5):
	concfeature = tf.keras.backend.concatenate(features, axis=1)
	w_fc_conc = tf.Variable(tf.random_normal(tf.TensorShape([concfeature.shape[1], tf.Dimension(2)]), stddev=0.1), trainable=training, dtype=tf.float32, name='w_fc_conc')
	b_fc_conc = tf.Variable([-math.log((1-positive_confidence)/positive_confidence), math.log((1-positive_confidence)/positive_confidence)], trainable=training, dtype=tf.float32, name='b_fc_conc')
	out_fc_conc = tf.add(tf.matmul(concfeature, w_fc_conc), b_fc_conc)
	out_sm = tf.nn.softmax(out_fc_conc)
	variables = {'w_fc_conc': w_fc_conc, 'b_fc_conc': b_fc_conc}
	return out_fc_conc, out_sm, variables