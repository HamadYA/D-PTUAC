from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = '/DATA/lcl/GOT/'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = '/DATA/LaSOTTest/'
    settings.network_path = '/home/ps/train-dimp/pytracking/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = '/home/ps/train-dimp/pytracking/pytracking/result_plots/'
    settings.results_path = '/home/user/zhuyabin/pytracking-tot2-gan-att-refine2-test2/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/ps/train-dimp/pytracking/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/media/ps/979e1f8d-4a13-40bb-8041-7e2e3a9a0b21/Trackingnet/'
    settings.uav_path = ''
    settings.vot_path = '/media/ps/979e1f8d-4a13-40bb-8041-7e2e3a9a0b21/VOT2018/'
    settings.youtubevos_dir = ''

    return settings

