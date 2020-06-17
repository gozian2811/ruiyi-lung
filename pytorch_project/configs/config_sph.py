import warnings
import os
from .config import BasicConfig
class DefaultConfig(BasicConfig):
    model = 'DenseCropNet_ParamWise'
    connection_mode = 'singleline'
    env = model + '_sph_test_fold1'
    Devices_ID = '1'
    
    dataset = 'SPH_Pathology_Dataset'
    label_mode = 'binclass'
    root_dir = '../data_samples/sph_cubes_64_overbound'
    filelist_shuffle = False
    num_cross_folds = 5
    val_fold = 1
    test_fold = -1
    filelist_train = BasicConfig.model_root + '/DensecropNet_classification_gr14_l4_2/filelist_train_fold[2, 3, 4].log'
    filelist_val = BasicConfig.model_root + '/DensecropNet_classification_gr14_l4_2/filelist_val_fold0.log'
    filelist_test = BasicConfig.model_root + '/DensecropNet_sph_3_fold1/filelist_val_fold0.log'
    
    #load_model_path = BasicConfig.model_root + '/ParamTest_sph_test_fold1/ParamTest_sph_test_fold1_epoch39'
    #load_model_path = "models_pytorch/DensecropNet_2_trand_fold1/DensecropNet_2_trand_fold1_epoch13"
    #load_model_path = "models_pytorch/DensecropNet_regression_2_fold1/DensecropNet_regression_2_fold1_epoch141"
    #load_model_path = "models_pytorch/DenseNet_Iterative_trand_fold5/DenseNet_Iterative_trand_fold5_epoch84"
    #load_model_path = "/home/lyj/zyplanet/DensecropNet/checkpoints/<class 'models.models.DensecropNet_Ini'>_0325_01:05:26.pkl"
    input_size = 64
    batch_size = 4
    print_freq = 5
    save_freq = 0
    
    translation_num = 1
    translation_range = (-3, 3)
    rotation_num = 1
    flip_num = 1
    noise_range = (-76, 76)

    num_blocks = 3
    growth_rate = 16
    num_final_growth = 1
    average_pool = False
    compression_factor = 0.5

    lr = 0.01
    #balancing = (0.5, 0.3, 0.2)
