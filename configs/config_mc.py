import warnings
import os
from .config import BasicConfig
class DefaultConfig(BasicConfig):
    model_root = '/data/fyl/models_pytorch'
    model = 'MulticropNet'
    model_setup = {'channels': [[64, 64, 64], [32]]}

    env = model + '_rfold1|10'
    Devices_ID = '1'
    
    dataset = 'SPH_Pathology_Dataset'
    label_mode = 'binclass'
    root_dir = '/data/fyl/data_samples/sph_cubes_64_overbound'
    #patientlist = '../data_samples/sph_overbound/patientlist.log'
    filelist_shuffle = False
    num_cross_folds = 10
    val_fold = 1
    test_fold = -1
    remove_uncertain = True
    filelists = dict(
        #train = model_root + '/DensecropNet_Iterative_Detection/filelist_train_fold[0, 1, 2, 3, 4, 5, 6, 7, 8].log'
        #val = model_root + '/DensecropNet_Iterative_Detection/filelist_val_fold9.log'
        test = model_root + '/MulticropNet_sph_classification_adam_rfold10|10/filelist_val_fold9.log'
    )
    
    load_model_path = None
    #load_model_path = model_root + '/MulticropNet_sph_classification_adam_rfold10|10/MulticropNet_sph_classification_adam_rfold10|10_epoch30'
    #load_model_path = "/home/lyj/zyplanet/DensecropNet/checkpoints/<class 'models.model.DenseNet'>_0325_05:07:57.pkl"
    #load_model_path = "/home/lyj/zyplanet/DensecropNet/checkpoints/<class 'models.models.DensecropNet_Denser'>_0325_01:22:59.pkl"
    #load_model_path = "/home/lyj/zyplanet/DensecropNet/checkpoints/<class 'models.models.DensecropNet'>_0324_18:04:27.pkl"
    #load_model_path = "/home/lyj/zyplanet/DensecropNet/checkpoints/<class 'models.models.DensecropNet_Ini'>_0325_01:05:26.pkl"
    input_size = 64
    batch_size = 4
    print_freq = 5
    save_freq = 5
    
    data_preprocess = dict(
        translation_num = 1,
        translation_range = (-3, 3),
        rotation_num = 1,
        flip_num = 1,
	shear_num = 0,
        noise_range = (-76, 76)
    )

    max_epoch = 300

    lr = 0.001
    
    loss_exp = 0
    balancing = (0.6, 0.4)
    drop_rate = 0
    weight_decay = 5e-4
