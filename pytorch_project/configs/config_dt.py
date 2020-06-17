import warnings
import os
from .config import BasicConfig
class DefaultConfig(BasicConfig):
    model_root = '/data/fyl/models_pytorch'
    model = 'DensecropNet'
    model_setup = dict(
        #connection_mode = 'singleline',
        num_blocks = 4,
        growth_rate = 64,
        num_fin_growth = 3,
        #avg_pool = False,
        drop_rate = 0.3
    )
    env = model + '_detection_2_rfold4'
    Devices_ID = '1'
    use_gpu = True

    dataset = 'Detection_Dataset'
    label_mode = 'binclass'
    root_dir = "/data/fyl/data_samples/luna_cubes_40_overbound"
    #root_dirs = ["/data/fyl/data_samples/tianchild_cubes_overbound/npy", "/data/fyl/data_samples/tianchild_cubes_overbound/npy_non", "/data/fyl/data_samples/tianchild_cubes_overbound/npy_non_fine"]
    patientlist = "/data/fyl/models_pytorch/DensecropNet_detection_2_rfold10/patientlist.log"
    filelist_shuffle = False
    num_cross_folds = 10
    val_fold = 4
    test_fold = -1
    filelist_easy = None
    filelists = dict(
        train = model_root + '/DensecropNet_Iterative_detection_5_fold6/filelist_train_fold[0, 1, 2, 3, 4, 6, 7, 8, 9].log',
        val = model_root + '/DensecropNet_Iterative_detection_5_fold6/filelist_val_fold5.log',
        test = model_root + '/DensecropNet_detection_rfold4/filelist_val_fold3.log'
    )

    #load_model_path = model_root + '/DensecropNet_Iterative_gr14_agp_fold3/DensecropNet_Iterative_gr14_agp_fold3_epoch165'
    load_model_path = "/data/fyl/models_pytorch/DensecropNet_detection_rfold1/DensecropNet_detection_rfold1_epoch0"
    #load_model_path = "/data/fyl/models_pytorch/DensecropNet_detection_rfold5/DensecropNet_detection_rfold5_epoch1"
    #load_model_path = "/home/lyj/zyplanet/DensecropNet/checkpoints/<class 'models.models.DensecropNet'>_0324_18:04:27.pkl"
    #load_model_path = "/home/lyj/zyplanet/DensecropNet/checkpoints/<class 'models.models.DensecropNet_Ini'>_0325_01:05:26.pkl"
    input_size = 32
    batch_size = 50
    print_freq = 100
    save_freq = 1
    save_filter = False

    data_preprocess = dict(
        translation_num = 1,
        translation_range = (-1, 1),
        rotation_num = 10,
        flip_num = 4,
        noise_range = (0, 0)
    )

    max_epoch = 30

    lr = 0.02
    
    loss_exp = 0
    balancing = (0.4, 0.6)
    weight_decay = 0

    compression_factor = 0.5
