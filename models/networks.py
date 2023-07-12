import paddle
import models.archs.classSR_rcan_arch as classSR_rcan_arch
import models.archs.HDR_arch as HDR_arch


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'HyCondITMv1':
        netG = HDR_arch.HyCondITMv1(transform_channels=opt_net['transform_channels'], global_cond_channels=opt_net['global_cond_channels'],
                              merge_cond_channels=opt_net['merge_cond_channels'], in_channels=opt_net['in_channels'])

    elif which_model == 'classSR_3class_rcan':
        netG = classSR_rcan_arch.classSR_3class_rcan(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG