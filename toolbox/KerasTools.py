from keras.models import Model, Sequential
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.layers import *
from keras import applications
from keras import backend as bk
from keras.engine.topology import Layer
import tensorflow as tf

class DesignedLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DesignedLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return bk.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def get_session(gpu_fraction=0.3):
	'''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

	num_threads = os.environ.get('OMP_NUM_THREADS')
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

	if num_threads:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
	else:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def vgg_implement(solid=False, partial=False, dropout_rate=0.0):
	model = Sequential()
	# Block 1
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224,224,3), name='block1_conv1'))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
	if dropout_rate>0:
		model.add(GaussianDropout(dropout_rate, name='block1_dropout'))

	# Block 2
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
	if dropout_rate>0:
		model.add(GaussianDropout(dropout_rate, name='block2_dropout'))

	# Block 3
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
	if dropout_rate>0:
		model.add(GaussianDropout(dropout_rate, name='block3_dropout'))

	# Block 4
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
	if dropout_rate>0:
		model.add(GaussianDropout(dropout_rate, name='block4_dropout'))

	# Block 5
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
	if dropout_rate>0:
		model.add(GaussianDropout(dropout_rate, name='block5_dropout'))

	# Classification block
	model.add(Flatten(name='flatten'))
	model.add(Dense(4096, activation='relu', name='fc1'))
	model.add(Dense(4096, activation='relu', name='fc2'))
	model.add(Dense(2, activation='softmax', name='fc3'))

	if solid:
		model.load_weights(filepath='models_keras/vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
	if partial:	#net partially trainable
		for layer in model.layers:
			if layer.name.find('fc')<0:
				layer.trainable = False
	return model

def VGG16(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
	"""Instantiates the VGG16 architecture.

	Optionally loads weights pre-trained
	on ImageNet. Note that when using TensorFlow,
	for best performance you should set
	`image_data_format="channels_last"` in your Keras config
	at ~/.keras/keras.json.

	The model and the weights are compatible with both
	TensorFlow and Theano. The data format
	convention used by the model is the one
	specified in your Keras config file.

	# Arguments
		include_top: whether to include the 3 fully-connected
			layers at the top of the network.
		weights: one of `None` (random initialization)
			or "imagenet" (pre-training on ImageNet).
		input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
			to use as image input for the model.
		input_shape: optional shape tuple, only to be specified
			if `include_top` is False (otherwise the input shape
			has to be `(224, 224, 3)` (with `channels_last` data format)
			or `(3, 224, 244)` (with `channels_first` data format).
			It should have exactly 3 inputs channels,
			and width and height should be no smaller than 48.
			E.g. `(200, 200, 3)` would be one valid value.
		pooling: Optional pooling mode for feature extraction
			when `include_top` is `False`.
			- `None` means that the output of the model will be
				the 4D tensor output of the
				last convolutional layer.
			- `avg` means that global average pooling
				will be applied to the output of the
				last convolutional layer, and thus
				the output of the model will be a 2D tensor.
			- `max` means that global max pooling will
				be applied.
		classes: optional number of classes to classify images
			into, only to be specified if `include_top` is True, and
			if no `weights` argument is specified.

	# Returns
		A Keras model instance.

	# Raises
		ValueError: in case of invalid argument for `weights`,
			or invalid input shape.
	"""
	if weights not in {'imagenet', None}:
		raise ValueError('The `weights` argument should be either '
				 '`None` (random initialization) or `imagenet` '
				 '(pre-training on ImageNet).')
	if weights == 'imagenet' and include_top and classes != 1000:
		raise ValueError('If using `weights` as imagenet with `include_top`'
				 ' as true, `classes` should be 1000')

	# Determine proper input shape
	input_shape = _obtain_input_shape(input_shape,
		                      default_size=224,
		                      min_size=48,
		                      data_format=K.image_data_format(),
		                      include_top=include_top)

	if input_tensor is None:
		img_input = Input(shape=input_shape)
	else:
		if not K.is_keras_tensor(input_tensor):
			img_input = Input(tensor=input_tensor, shape=input_shape)
		else:
			img_input = input_tensor
	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

	if include_top:
		# Classification block
		x = Flatten(name='flatten')(x)
		x = Dense(4096, activation='relu', name='fc1')(x)
		x = Dense(4096, activation='relu', name='fc2')(x)
		x = Dense(classes, activation='softmax', name='predictions')(x)
	else:
		if pooling == 'avg':
			x = GlobalAveragePooling2D()(x)
		elif pooling == 'max':
			x = GlobalMaxPooling2D()(x)

	# Ensure that the model takes into account
	# any potential predecessors of `input_tensor`.
	if input_tensor is not None:
		inputs = get_source_inputs(input_tensor)
	else:
		inputs = img_input
	# Create model.
	model = Model(inputs, x, name='vgg16')

	# load weights
	weights_path = 'models_keras/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
	weights_path_no_top = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
	if weights == 'imagenet':
		if include_top:
			weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
					    weights_path,
					    cache_subdir='models')
		else:
			weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
					    weights_path_no_top,
					    cache_subdir='models')
		model.load_weights(weights_path)
		if K.backend() == 'theano':
			layer_utils.convert_all_kernels_in_model(model)

		if K.image_data_format() == 'channels_first':
			if include_top:
				maxpool = model.get_layer(name='block5_pool')
				shape = maxpool.output_shape[1:]
				dense = model.get_layer(name='fc1')
				layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

			if K.backend() == 'tensorflow':
				warnings.warn('You are using the TensorFlow backend, yet you '
					      'are using the Theano '
					      'image data format convention '
					      '(`image_data_format="channels_first"`). '
					      'For best performance, set '
					      '`image_data_format="channels_last"` in '
					      'your Keras config '
					      'at ~/.keras/keras.json.')
	return model
