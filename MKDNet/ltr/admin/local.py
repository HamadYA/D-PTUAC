class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.lasot_dir = '/DATA/lcl/LASOT/LASOT/'
        self.got10k_dir = '/DATA/lcl/GOT/train/'
        self.trackingnet_dir = '/DATA/lcl/TrackingNet/'
        self.coco_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.sot_dir = '/DATA/lcl/SOT_TRAIN/'
        #self.sot_dir = '/media/ps/979e1f8d-4a13-40bb-8041-7e2e3a9a0b21/SOT-TRAIN/SOT/'