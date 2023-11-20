import math
import torch
import torch.nn as nn
from collections import OrderedDict
from ltr.models.meta import steepestdescent
import ltr.models.target_classifier.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.initializer as clf_initializer
import ltr.models.target_classifier.optimizer as clf_optimizer
import ltr.models.target_classifier.networks_d as define_d
import ltr.models.bbreg as bbmodels
import ltr.models.backbone as backbones
from ltr import model_constructor
from ltr.admin import loading
import pdb
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from pylab import *
import cv2
import random
import pdb
import imageio
import collections
from torchvision.transforms import ToPILImage
import numpy as np
from skimage.transform import resize
import torch.nn.functional as F
class DiMPnet(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    def __init__(self, feature_extractor_t,feature_extractor, classifier,classifier_t, bb_regressor,bb_regressor_t, classification_layer,feat_D1,feat_D2, bb_regressor_layer):
        super().__init__()
        self.feature_extractor_t = feature_extractor_t
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.classifier_t = classifier_t        
        self.bb_regressor = bb_regressor
        self.bb_regressor_t = bb_regressor_t        
        self.feat_D1=feat_D1
        self.feat_D2=feat_D2        
        self.classification_layer = [classification_layer] if isinstance(classification_layer, str) else classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.output_layers = sorted(list(set(self.classification_layer + self.bb_regressor_layer)))
        self.l1_loss = torch.nn.L1Loss()
        
    def forward(self, train_imgs, test_imgs, train_bb, test_bb, test_proposals, *args, **kwargs):
        """Runs the DiMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            test_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module.
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            iou_pred:  Predicted IoU scores for the test_proposals."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        #print(train_imgs.shape)
        [i,s,c]=train_bb.shape 
        r=i*s

        
        sr_train=((train_bb.reshape(-1,4)).sum(dim=0))/r/16
        sr_test=((test_bb.reshape(-1,4)).sum(dim=0))/r/16   
        '''img=img.resize((new_w,new_h),Image.BICUBIC)

        if random.uniform(0,1)<0.5:
            img=img.resize((w,h),Image.NEAREST)
        else:
            img = img.resize((w, h), Image.BILINEAR)''' 
        train_imgs_mid=F.interpolate(train_imgs.reshape(-1, *train_imgs.shape[-3:]), [int(352/sr_train[2]),int(352/sr_train[3])], mode='bicubic')  
        test_imgs_mid=F.interpolate(test_imgs.reshape(-1, *test_imgs.shape[-3:]), [int(352/sr_test[2]),int(352/sr_test[3])], mode='bicubic') 
        if random.uniform(0,1)<0.5:
            q=nn.Upsample(size=(352,352), mode='nearest')
        else:
            q=nn.Upsample(size=(352,352),mode='bilinear')        
        train_imgs_d=q(train_imgs_mid)
        test_imgs_d=q(test_imgs_mid)
        '''plt.figure() 
        img=train_imgs[0,0,:,:,:].cpu().numpy()
        #img=(img).astype(np.uint8)         
        img=img.reshape(img.shape[1],img.shape[2],img.shape[0])
        np.save("filename.npy",img)
        #x=np.load("/home/lcl/zhuyabin/pytracking/ltr/filename.npy")
        plt.imshow(img)
        
        plt.title("oral")
        plt.show()
        axis('off')        
        img=train_imgs_d[0,0,:,:,:].cpu().numpy()
        #img=(img).astype(np.uint8)         
        img=img.reshape(img.shape[1],img.shape[2],img.shape[0]) 
        plt.imshow(img) 
        plt.title("2")        
        plt.show()
        axis('off')         
        #cv2.imshow("img",x)
        #cv2.waitKey(0)'''

        # Extract backbone features
        train_feat_t = self.extract_backbone_features_t(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat_t = self.extract_backbone_features_t(test_imgs.reshape(-1, *test_imgs.shape[-3:]))
        train_feat_s = self.extract_backbone_features(train_imgs_d.reshape(-1, *train_imgs.shape[-3:]))
        test_feat_s = self.extract_backbone_features(test_imgs_d.reshape(-1, *test_imgs.shape[-3:]))
        #pdb.set_trace()   




        
        # Classification feature
        train_feat_clf = self.get_backbone_clf_feat(train_feat_s)
        test_feat_clf = self.get_backbone_clf_feat(test_feat_s)
        train_feat_clf_t = self.get_backbone_clf_feat(train_feat_t)
        test_feat_clf_t = self.get_backbone_clf_feat(test_feat_t)
        # Run classifier module
        target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb, *args, **kwargs)
        target_scores_t = self.classifier_t(train_feat_clf_t, test_feat_clf_t, train_bb, *args, **kwargs)
        # Get bb_regressor features
        train_feat_iou = self.get_backbone_bbreg_feat(train_feat_s)
        test_feat_iou = self.get_backbone_bbreg_feat(test_feat_s)
        train_feat_iou_t = self.get_backbone_bbreg_feat(train_feat_t)
        test_feat_iou_t = self.get_backbone_bbreg_feat(test_feat_t)        
        #pdb.set_trace()
        # Run the IoUNet module
        iou_pred,roi3t,roi4t = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)
        iou_pred_t,roi3t_t,roi4t_t = self.bb_regressor_t(train_feat_iou_t, test_feat_iou_t, train_bb, test_proposals)
        #pdb.set_trace()        
        #L2,Discreiminator
        pred_fake_feat_d1=self.feat_D1.forward(roi3t.detach())
        pred_real_feat_d1=self.feat_D1.forward(roi3t_t.detach())
        #L2,Generator
        pred_fake_feat_G1 =self.feat_D1.forward(roi3t)

        #L3,Discreiminator
        pred_fake_feat_d2=self.feat_D2(roi4t.detach())
        pred_real_feat_d2=self.feat_D2.forward(roi4t_t.detach())
        #L3,Generator
        pred_fake_feat_G2 =self.feat_D2.forward(roi4t)        

        return target_scores,target_scores_t, iou_pred,iou_pred_t,pred_fake_feat_d1,pred_real_feat_d1,pred_fake_feat_G1,pred_fake_feat_d2,pred_real_feat_d2,pred_fake_feat_G2,roi3t,roi4t,roi3t_t,roi4t_t
        #return target_scores, iou_pred, kd_loss, loss_featD, loss_G, test_loss_featD, test_loss_G

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)
    def extract_backbone_features_t(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor_t(im, layers)
    def extract_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer + ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})



@model_constructor
def dimpnet18(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=1,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=256, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
              mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              score_act='relu', act_param=None, target_mask_act='sigmoid',
              detach_length=float('Inf'), frozen_backbone_layers=()):
    # Backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def dimpnet50(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=512, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
              mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              score_act='relu', act_param=None, target_mask_act='sigmoid',
              detach_length=float('Inf'), frozen_backbone_layers=()):

    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    backbone_net_t = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if classification_layer == 'layer3':
        feature_dim = 256
    elif classification_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)
    classifier_t = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)
    # Bounding box regressor
    bb_regressor_t = bbmodels.AtomIoUNet(input_dim=(4*128,4*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4*128,4*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)    
# load pretrained model
    pretrainmodel_path='/home/lcl/zhuyabin/pytracking/ltr/networks/super_dimp.pth.tar'
    pretrainmodel = loading.torch_load_legacy(pretrainmodel_path)['net']
    
    #******************train******************#
    usepretrain = updback = updcls = updbb = True 
    #**************test*******************#
    usepretrain = updback = updcls = updbb = False    
    if usepretrain:
        print('pretrained model path', pretrainmodel_path)
        if updback:
            # update backbone
            backbone_dict = backbone_net.state_dict()
            pretrain_dict = {k[len('feature_extractor.'):]: v for k, v in pretrainmodel.items() if k[len('feature_extractor.'):] in backbone_dict}
            backbone_net.load_state_dict(pretrain_dict)
            backbone_net_t.load_state_dict(pretrain_dict)            
        if updcls:
            # update classifier
            classifier_dict = classifier.state_dict()
            pretrain_dict = {k[len('classifier.'):]: v for k, v in pretrainmodel.items() if k[len('classifier.'):] in classifier_dict}
            classifier.load_state_dict(pretrain_dict)
            classifier_t.load_state_dict(pretrain_dict)            
        if updbb:
            # update Bounding box regressor 
            bb_regressor_dict = bb_regressor.state_dict()
            pretrain_dict = {k[len('bb_regressor.'):]: v for k, v in pretrainmodel.items() if k[len('bb_regressor.'):] in bb_regressor_dict}
            bb_regressor.load_state_dict(pretrain_dict)
            bb_regressor_t.load_state_dict(pretrain_dict)            
        print('load pretrained model end!')  
    feat_D1=define_d.Discriminator1() 
    feat_D2=define_d.Discriminator2()                                          
    # DiMP network
    net = DiMPnet(feature_extractor_t=backbone_net_t,feature_extractor=backbone_net, classifier=classifier,classifier_t=classifier_t, bb_regressor=bb_regressor, bb_regressor_t=bb_regressor_t,
                  classification_layer=classification_layer,feat_D1=feat_D1,feat_D2=feat_D2, bb_regressor_layer=['layer2', 'layer3'])
                  
    return net



@model_constructor
def L2dimpnet18(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=1,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=256, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              detach_length=float('Inf'), hinge_threshold=-999, gauss_sigma=1.0, alpha_eps=0):
    # Backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPL2SteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step, hinge_threshold=hinge_threshold,
                                                    init_filter_reg=optim_init_reg, gauss_sigma=gauss_sigma,
                                                    detach_length=detach_length, alpha_eps=alpha_eps)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def klcedimpnet18(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
                  classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=1,
                  clf_feat_norm=True, init_filter_norm=False, final_conv=True,
                  out_feature_dim=256, gauss_sigma=1.0,
                  iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
                  detach_length=float('Inf'), alpha_eps=0.0, train_feature_extractor=True,
                  init_uni_weight=None, optim_min_reg=1e-3, init_initializer='default', normalize_label=False,
                  label_shrink=0, softmax_reg=None, label_threshold=0, final_relu=False, init_pool_square=False,
                  frozen_backbone_layers=()):

    if not train_feature_extractor:
        frozen_backbone_layers = 'all'

    # Backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim, final_relu=final_relu)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim, init_weights=init_initializer,
                                                          pool_square=init_pool_square)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.PrDiMPSteepestDescentNewton(num_iter=optim_iter, feat_stride=feat_stride,
                                                          init_step_length=optim_init_step,
                                                          init_filter_reg=optim_init_reg, gauss_sigma=gauss_sigma,
                                                          detach_length=detach_length, alpha_eps=alpha_eps,
                                                          init_uni_weight=init_uni_weight,
                                                          min_filter_reg=optim_min_reg, normalize_label=normalize_label,
                                                          label_shrink=label_shrink, softmax_reg=softmax_reg,
                                                          label_threshold=label_threshold)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def klcedimpnet50(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
                  classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
                  clf_feat_norm=True, init_filter_norm=False, final_conv=True,
                  out_feature_dim=512, gauss_sigma=1.0,
                  iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
                  detach_length=float('Inf'), alpha_eps=0.0, train_feature_extractor=True,
                  init_uni_weight=None, optim_min_reg=1e-3, init_initializer='default', normalize_label=False,
                  label_shrink=0, softmax_reg=None, label_threshold=0, final_relu=False, frozen_backbone_layers=()):

    if not train_feature_extractor:
        frozen_backbone_layers = 'all'

    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_bottleneck(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim, final_relu=final_relu)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim, init_weights=init_initializer)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.PrDiMPSteepestDescentNewton(num_iter=optim_iter, feat_stride=feat_stride,
                                                          init_step_length=optim_init_step,
                                                          init_filter_reg=optim_init_reg, gauss_sigma=gauss_sigma,
                                                          detach_length=detach_length, alpha_eps=alpha_eps,
                                                          init_uni_weight=init_uni_weight,
                                                          min_filter_reg=optim_min_reg, normalize_label=normalize_label,
                                                          label_shrink=label_shrink, softmax_reg=softmax_reg,
                                                          label_threshold=label_threshold)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4*128,4*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net
