import warnings
import os
from .config import BasicConfig
class DefaultConfig(BasicConfig):
    model_root = '/data/fyl/models_pytorch'
    model = 'ParamDenseCropNet'
    model_setup = dict(
        num_blocks = 4,
        growth_rate = 20,
        num_fin_growth = 1,
        #avg_pool = False,
	drop_rate = 0.15
    )

    env = model + '_diamregression_sph'
    Devices_ID = '0'
    
    dataset = 'SPH_Pathology_Dataset'
    label_mode = None
    root_dir = '/data/fyl/data_samples/sph_overbound'
    #root_dir = model_root + "/BasicNet_lidc_classification_9/filelist_train_fold[0, 1, 2, 3, 4].log"
    #label_collection_file = 'statistics_3d.csv'
    #patientlist = 'models_pytorch/ParamDenseCropNet_diameter_lidc_rfold1/patientlist.log'
    filelist_shuffle = False
    num_cross_folds = 5
    val_fold = -1
    test_fold = -1
    remove_uncertain = True
    filelists = dict(
        train = model_root + '/ParamDenseCropNet_diameter_lidc_rfold1/filelist_train_fold[1, 2, 3, 4].log',
        val = model_root + '/ParamDenseCropNet_diameter_lidc_rfold1/filelist_val_fold0.log',
        test = model_root + '/ParamDenseCropNet_diamregression_sph/filelist_train_fold[0, 1, 2, 3, 4].log'
    )
    
    #load_model_path = BasicConfig.model_root + "/MultiTaskNet_classification_2_fold1/MultiTaskNet_classification_2_fold1_epoch400"
    #load_model_path = model_root + '/BasicNet_lidc_classification_9/BasicNet_lidc_classification_9_epoch500'
    #load_model_path = model_root + '/BasicNet_classification_lidc_transfer_rfold5/BasicNet_classification_lidc_transfer_rfold5_epoch1330'
    #load_model_path = "models_pytorch/DensecropNet_regression_2_fold1/DensecropNet_regression_2_fold1_epoch141"
    #load_model_path = model_root + "/DenseNet_Iterative_trand_fold1/DenseNet_Iterative_trand_fold1_epoch101"
    load_model_path = model_root + "/ParamDenseCropNet_diamregression_2_lidc_rfold1/ParamDenseCropNet_diamregression_2_lidc_rfold1_epoch1030"
    transfer = False

    input_size = 64
    batch_size = 10
    print_freq = 5
    save_freq = 10
    
    data_preprocess = dict(
        translation_num = 1,
        translation_range = (-3, 3),
        rotation_num = 1,
        flip_num = 1,
	shear_num = 1,
        noise_range = (-76, 76)
    )

    loss_exp = 0
    balancing = (0.6, 0.4)

    compression_factor = 0.5

    max_epoch = 1500
    lr = 1e-4
