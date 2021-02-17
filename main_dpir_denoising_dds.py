import os.path
import logging

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util

from skimage.restoration import estimate_sigma


"""
Spyder (Python 3.7)
PyTorch 1.6.0
Windows 10 or Linux
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/DPIR
        https://github.com/cszn/IRCNN
        https://github.com/cszn/KAIR
@article{zhang2020plug,
  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint},
  year={2020}
}
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; homepage: https://cszn.github.io/)
by Kai Zhang (01/August/2020)

# --------------------------------------------
|--model_zoo               # model_zoo
   |--drunet_gray          # model_name, for color images
   |--drunet_color
|--testset                 # testsets
   |--set12                # testset_name
   |--bsd68
   |--cbsd68
|--results                 # results
   |--set12_dn_drunet_gray # result_name = testset_name + '_' + 'dn' + model_name
   |--set12_dn_drunet_color
# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    #noise_level_img = 50                 # default: 15, set AWGN noise level for noisy image
    #noise_level_model = noise_level_img  # set noise level for model
    model_name = 'drunet_color'           # set denoiser model, 'drunet_gray' | 'drunet_color'
    # 'bsd68'               # set test set,  'bsd68' | 'cbsd68' | 'set12'
    testset_name = 'g150_e400000' #'21_Sep_2020_Field_test'
    x8 = False                           # default: False, x8 to boost performance
    show_img = False                     # default: False
    border = 0                           # shave boader to calculate PSNR and SSIM

    if 'color' in model_name:
        n_channels = 3                   # 3 for color image
    else:
        n_channels = 1                   # 1 for grayscale image

    model_pool = 'model_zoo'             # fixed
    testsets = 'testsets'                # fixed
    results = 'results'                  # fixed
    task_current = 'dn'                  # 'dn' for denoising
    result_name = testset_name + '_' + task_current + '_' + model_name

    model_path = os.path.join(model_pool, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using = ", device)
    #torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_unet import UNetRes as net
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    logger.info('model_name:{}'.format(model_name))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        imgInput = util.read_img(img)
        imgOrg = imgInput
        imgNoiseLevel = estimate_sigma(imgInput, multichannel=True, average_sigmas=True) * 9
        logger.info(f'image noise nevel: {imgNoiseLevel}')
        
        #img_H = util.imread_uint(img, n_channels=n_channels)
        # sigma_est = estimate_sigma(img_H, multichannel=True, average_sigmas=True)
        #img_L = util.uint2single(img_H)

        # # Add noise without clipping
        # np.random.seed(seed=0)  # for reproducibility
        # img_L += np.random.normal(0, noise_level_img/255., img_L.shape)

        util.imshow(util.single2uint(imgInput), title='Noisy image with noise level {}'.format(imgNoiseLevel*255.0)) if show_img else None

        imgInput = util.single2tensor4(imgInput)
        imgInput = torch.cat((imgInput, torch.FloatTensor([imgNoiseLevel]).repeat(1, 1, imgInput.shape[2], imgInput.shape[3])), dim=1)
        imgInput = imgInput.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------
        img_E = model(imgInput)
        # if not x8 and imgInput.size(2)//8==0 and imgInput.size(3)//8==0:
        #     img_E = model(imgInput)
        # elif not x8 and (imgInput.size(2)//8!=0 or imgInput.size(3)//8!=0):
        #     img_E = utils_model.test_mode(model, imgInput, refield=64, mode=5)
        # elif x8:
        #     img_E = utils_model.test_mode(model, imgInput, mode=3)

        img_E = util.tensor2uint(img_E)

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------

        if n_channels == 1:
            imgOrg = imgOrg.squeeze() 
        psnr = util.calculate_psnr(img_E, imgOrg, border=border)
        ssim = util.calculate_ssim(img_E, imgOrg, border=border)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))

        # ------------------------------------
        # save results
        # ------------------------------------

        util.imsave(img_E, os.path.join(E_path, img_name+ext))

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))


if __name__ == '__main__':

    main()
