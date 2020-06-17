import warnings
import os
from .config import BasicConfig
class DefaultConfig(BasicConfig):
    model_root = '/data/fyl/models_pytorch'
    model = 'BaseweightSharedBNNet'
    model_setup = dict(
        normps = (2, 1, 1),
        #num_blocks = 4,
        #growth_rate = 20,
        #num_final_growth = 1,
        #average_pool = False,
        #drop_rate = 0.15
        pretrained = True
    )

    env = model + '_classification_pretrained_oa_3slc_adam_3_rfold1|10'
    Devices_ID = '4'
    use_gpu = True

    dataset = 'Pathology_Slicewise_Dataset'
    dataset2 = 'Pathology_Slicewise_Dataset'
    label_mode = 'binclass'
    patch_mode = 'oa'
    patch_mode2 = 'oa'
    root_dir = '/data/fyl/data_samples/lidc_3slices'
    #root_dir = BasicConfig.model_root + "/BasicNet_lidc_classification_9/filelist_train_fold[0, 1, 2, 3, 4].log"
    root_dir2 = '/data/fyl/data_samples/sph_overbound'
    #root_dir2 = BasicConfig.model_root + "/BasicNet_sph_classification_8/filelist_train_fold[0, 1, 2, 3, 4].log"
    patientlist = '/data/fyl/models_pytorch/BaseweightSharedBNNet_classification_pretrained_adam_rfold8|10/patientlist.log'
    patientlist2 = '/data/fyl/models_pytorch/BaseweightSharedBNNet_classification_pretrained_adam_rfold8|10/patientlist2.log'
    #patientlist2 = None
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
        test = model_root + '/BaseweightSharedBNNet_classification_pretrained_oa_3slc_adam_rfold1|10/filelist_val_fold0.log'
    )
    filelists2 = dict(
        train = model_root + '/BaseweightSharedNet_classification_2_rfold3|10/filelist2_train_fold[0, 1, 3, 4, 5, 6, 7, 8, 9].log',
        val = model_root + '/BaseweightSharedNet_classification_2_rfold3|10/filelist2_val_fold2.log',
        test = model_root + '/BaseweightSharedBNNet_classification_pretrained_oa_3slc_adam_rfold1|10/filelist2_val_fold0.log'
    )

    task_bias = -1
    #load_model_path = None
    #load_model_path = BasicConfig.model_root + "/MultiTaskNet_classification_2_fold1_section1/MultiTaskNet_classification_2_fold1_section1_epoch16"
    #load_model_path = model_root + "/BaseweightSharedBNNet_bg1_classification_adam_rfold1|10/BaseweightSharedBNNet_bg1_classification_adam_rfold1|10_epoch26"
    #load_model_path = model_root + "/BaseweightSharedBNNet_classification_scratch_oa_adam_rfold8|10/BaseweightSharedBNNet_classification_scratch_oa_adam_rfold8|10_epoch16"
    #load_model_path = model_root + "/RegularlySharedNet_classification_blidc_rfold1/RegularlySharedNet_classification_blidc_rfold1_epoch1500"
    #load_model_path = model_root + '/BasicNet_sph_classification_8_fold3/BasicNet_sph_classification_8_fold3_epoch335'
    #load_model_path = model_root + "/BasicNet_lidc_classification_9/BasicNet_lidc_classification_9_epoch1500"
    inference_setup = {'model_name': 'BaseweightSharedBNNet_classification_pretrained_oa_3slc_adam_2_rfold1|10', 'epochs': [i for i in range(1, 20)]}
    input_size = 200
    batch_size = 60
    print_freq = 5
    save_freq = 1

    data_preprocess = dict(
        translation_num = 1,
        translation_range = (-10, 10),
        rotation_num = 3,
        flip_num = 2,
	shear_num = 0,
        noise_range = (0, 0)
    )

    loss_weightlist = (0.5, 0.5, 7.4e-4, 7.4e-4, 7.4e-4)
    loss_exp = 0
    balancing = (0.5, 0.5)
    weight_decay = 0

    compression_factor = 0.5

    #the value of param_train_mode lies in {'overall', 'separate', 'sharing', 'inturn'}
    param_train_mode = 'overall'
    max_epoch = 60
    lr = 1e-5
    #lr_changes = [(100, 0.01)]
