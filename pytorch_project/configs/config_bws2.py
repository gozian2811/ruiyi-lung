import os
from .config import BasicConfig
class DefaultConfig(BasicConfig):
    model_root = '/data/fyl/models_pytorch'
    model = 'MultiTaskNet'
    model_setup = dict(
        #normps = (2, 1, 2),
        num_blocks = 4,
        growth_rate = 20,
        num_fin_growth = 1,
        #average_pool = False,
	drop_rate = 0.15
	#pretrained = True
    )

    env = model + '_densecropnet_classification_adam_rfold1|10'
    Devices_ID = '2'
    use_gpu = False

    dataset = 'LIDC_Pathology_Dataset'
    dataset2 = 'SPH_Pathology_Dataset'
    label_mode = 'binclass'
    #patch_mode = 'ensemble'
    #patch_mode2 = 'oa'
    root_dir = '/data/fyl/data_samples/lidc_cubes_64_overbound'
    #root_dir = BasicConfig.model_root + "/BasicNet_lidc_classification_9/filelist_train_fold[0, 1, 2, 3, 4].log"
    root_dir2 = '/data/fyl/data_samples/sph_overbound'
    #root_dir2 = BasicConfig.model_root + "/BasicNet_sph_classification_8/filelist_train_fold[0, 1, 2, 3, 4].log"
    patientlist = '/data/fyl/models_pytorch/BaseweightSharedNet_classification_2_rfold10|10/patientlist.log'
    #patientlist2 = '/data/fyl/models_pytorch/EnsembleNet_Paramwise_sph_classification_pretrained_adam_rfold1|10/patientlist.log'
    patientlist2 = None
    filelist_shuffle = False
    num_cross_folds = 10
    val_fold = 1
    test_fold = -1
    val_fold2 = 1
    test_fold2 = -1
    remove_uncertain = True
    filelists = dict(
        train = model_root + '/BaseweightSharedNet_classification_2_rfold3|10/filelist_train_fold[0, 1, 3, 4, 5, 6, 7, 8, 9].log',
        val = model_root + '/BaseweightSharedNet_classification_2_rfold3|10/filelist_val_fold2.log',
        test = model_root + '/BaseweightSharedNet_densecropnet_classification_adam_rfold1|10/filelist.log'
    )
    filelists2 = dict(
        train = model_root + '/BaseweightSharedNet_classification_2_rfold3|10/filelist2_train_fold[0, 1, 3, 4, 5, 6, 7, 8, 9].log',
        val = model_root + '/BaseweightSharedNet_classification_2_rfold3|10/filelist2_val_fold2.log',
        test = model_root + '/BaseweightSharedNet_densecropnet_classification_adam_rfold1|10/filelist2.log'
    )

    task_bias = -1
    #load_model_path = None
    load_model_path = model_root + "/MultiTaskNet_classification_3_fold1/MultiTaskNet_classification_3_fold1_epoch900"
    #load_model_path = model_root + "/BaseweightSharedBNNet_bg1_classification_adam_rfold1|10/BaseweightSharedBNNet_bg1_classification_adam_rfold1|10_epoch26"
    #load_model_path = model_root + "/BaseweightSharedBNNet_classification_adam_rfold7|10/BaseweightSharedBNNet_classification_adam_rfold7|10_epoch37"
    #load_model_path = model_root + "/RegularlySharedNet_classification_blidc_rfold1/RegularlySharedNet_classification_blidc_rfold1_epoch1500"
    #load_model_path = model_root + '/BasicNet_sph_classification_8_fold3/BasicNet_sph_classification_8_fold3_epoch335'
    #load_model_path = model_root + "/BasicNet_lidc_classification_9/BasicNet_lidc_classification_9_epoch1500"
    inference_setup = {'model_name': 'BaseweightSharedBNNet_classification_adam_rfold10|10', 'epochs': [1, 3, 4, 5, 8]}
    input_size = 64
    batch_size = 20
    print_freq = 5
    save_freq = 1

    data_preprocess = dict(
        translation_num = 1,
        translation_range = (-2, 2),
        rotation_num = 1,
        flip_num = 1,
	shear_num = 0,
        noise_range = (-76, 76)
    )

    loss_weightlist = (0.5, 0.5, 7.4e-4, 7.4e-4, 7.4e-4)
    loss_exp = 0
    balancing = (0.6, 0.4)
    weight_decay = 0

    compression_factor = 0.5

    #the value of param_train_mode lies in {'overall', 'separate', 'sharing', 'inturn'}
    param_train_mode = 'overall'
    max_epoch = 1500
    lr = 1e-4
    #lr_changes = [(100, 0.01)]
