from . import BaseActor
import torch
import collections
import pdb
class DiMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, iou_pred = self.net(train_imgs=data['train_images'],
                                           test_imgs=data['test_images'],
                                           train_bb=data['train_anno'],
                                           test_proposals=data['test_proposals'])

        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Compute loss for ATOM IoUNet
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred, data['proposal_iou'])

        # Loss for the initial filter iteration
        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a*b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # Total loss
        loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item()}
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        stats['ClfTrain/test_loss'] = clf_loss_test.item()
        if len(clf_losses_test) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
            if len(clf_losses_test) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        return loss, stats


class KLDiMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        #pdb.set_trace()        
        # Run network
        #pdb.set_trace()
        #, kd_loss, loss_featD, loss_G, test_loss_featD, test_loss_G
        target_scores,target_scores_t, bb_scores, bb_scores_t, pred_fake_feat_d1,pred_real_feat_d1,pred_fake_feat_G1,\
        pred_fake_feat_d2,pred_real_feat_d2,pred_fake_feat_G2,roi3t,roi4t,roi3t_t,roi4t_t = self.net(train_imgs=data['train_images'],
                                                           test_imgs=data['test_images'],                                          
                                                           train_bb=data['train_anno'],
                                                           test_bb=data['test_anno'],
                                                           test_proposals=data['test_proposals'])
        #pdb.set_trace() 
        [b,p]=pred_fake_feat_d1.shape                                                   
        target_true = torch.ones([b*p,1]).cuda()
        target_false = torch.zeros([b*p,1]).cuda()        
        kd1 =self.objective['l1_loss'](roi3t,roi3t_t)
        kd2 = self.objective['l1_loss'](roi4t,roi4t_t)        
        #kd_loss=self.loss_weight['criterionGAN']*(kd1+kd2)
        #pdb.set_trace()
        loss_featD_fake1 = self.objective['criterionGAN'](pred_fake_feat_d1.reshape(-1,1), target_false)
        loss_featD_real1 = self.objective['criterionGAN'](pred_real_feat_d1.reshape(-1,1), target_true)
        loss_G_featD1 =self.objective['criterionGAN'](pred_fake_feat_G1.reshape(-1,1),target_true)
        loss_featD_fake2 = self.objective['criterionGAN'](pred_fake_feat_d2.reshape(-1,1), target_false) 
        loss_featD_real2 = self.objective['criterionGAN'](pred_real_feat_d2.reshape(-1,1), target_true)
        loss_G_featD2 =self.objective['criterionGAN'](pred_fake_feat_G2.reshape(-1,1),target_true)    
        

                                                           
        #pdb.set_trace()
        #pdb.set_trace()
        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        bb_scores_t = bb_scores_t[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce        
        bb_ce_t = self.objective['bb_ce'](bb_scores_t, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        y=-torch.ones(1).cuda()
        loss_bbce_d=self.objective['max'](bb_ce.view(1,1),bb_ce_t.view(1,1),y)
        kd_loss=self.loss_weight['kd'] *loss_bbce_d*(kd1+kd2)
        loss_featD = (loss_featD_fake1 + loss_featD_real1 + loss_featD_fake2 + loss_featD_real2)*0.25
        loss_G = 2*loss_bbce_d*(loss_G_featD1 + loss_G_featD2)*0.5        

        #print("gggggggggg",loss_bbce_d,kd_loss,loss_G)
        # If standard DiMP classifier is used
        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        if 'test_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

            # Loss of the final filter
            clf_loss_test = clf_losses_test[-1]
            loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                else:
                    loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])
            #***********************************teacher cls_loss  **************************#        
            # Classification losses for the different optimization iterations
            clf_losses_test_t = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores_t]

            # Loss of the final filter
            clf_loss_test_t = clf_losses_test_t[-1]
            loss_target_classifier_t = self.loss_weight['test_clf'] * clf_loss_test_t

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf_t = self.loss_weight['test_init_clf'] * clf_losses_test_t[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf_t = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test_t[1:-1])])
                else:
                    loss_test_iter_clf_t = (test_iter_weights / (len(clf_losses_test_t) - 2)) * sum(clf_losses_test_t[1:-1])                    
                    
                    

        # If PrDiMP classifier is used
        loss_clf_ce = 0
        loss_clf_ce_init = 0
        loss_clf_ce_iter = 0
        if 'clf_ce' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_ce_losses = [self.objective['clf_ce'](s, data['test_label_density'], grid_dim=(-2,-1)) for s in target_scores]

            # Loss of the final filter
            clf_ce = clf_ce_losses[-1]
            loss_clf_ce = self.loss_weight['clf_ce'] * clf_ce

            # Loss for the initial filter iteration
            if 'clf_ce_init' in self.loss_weight.keys():
                loss_clf_ce_init = self.loss_weight['clf_ce_init'] * clf_ce_losses[0]

            # Loss for the intermediate filter iterations
            if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
                test_iter_weights = self.loss_weight['clf_ce_iter']
                if isinstance(test_iter_weights, list):
                    loss_clf_ce_iter = sum([a * b for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])])
                else:
                    loss_clf_ce_iter = (test_iter_weights / (len(clf_ce_losses) - 2)) * sum(clf_ce_losses[1:-1])


        cls_loss=loss_target_classifier + loss_test_init_clf + loss_test_iter_clf
        cls_loss_t=loss_target_classifier_t + loss_test_init_clf_t + loss_test_iter_clf_t
        #pdb.set_trace()
        y=-torch.ones(1).cuda()
        loss_cls_d=self.objective['max'](cls_loss.view(1,1),cls_loss_t.view(1,1),y)       
        imitation_loss_cls=4*loss_cls_d*(self.objective['l1_loss'](target_scores[0],target_scores_t[0])+self.objective['l1_loss'](target_scores[1],target_scores_t[1]))
        imitation_loss_reg=0.05*loss_bbce_d*self.objective['l1_loss'](bb_scores,bb_scores_t)
        #print("imi-------------",imitation_loss_cls,imitation_loss_reg)        
        #cls_loss-cls_loss_t
        # Total loss
        loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
                            loss_target_classifier + loss_test_init_clf + loss_test_iter_clf + kd_loss+loss_G + imitation_loss_cls + imitation_loss_reg
        loss_D=loss_featD
        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')
        #print('Loss/loss_bbce_d',loss_bbce_d.item())
        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bbce_d': loss_bbce_d.item(),                 
                 'Loss/kd_loss': kd_loss.item(),
                 'Loss/imitation_loss_cls': imitation_loss_cls.item(),
                 'Loss/imitation_loss_reg': imitation_loss_reg.item(),                 
                 'Loss/loss_featD': loss_featD.item(), 
                 'Loss/loss_G': loss_G.item(),                   
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        if 'clf_ce' in self.loss_weight.keys():
            stats['Loss/clf_ce'] = loss_clf_ce.item()
        if 'clf_ce_init' in self.loss_weight.keys():
            stats['Loss/clf_ce_init'] = loss_clf_ce_init.item()
        if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            stats['Loss/clf_ce_iter'] = loss_clf_ce_iter.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        if 'clf_ce' in self.loss_weight.keys():
            stats['ClfTrain/clf_ce'] = clf_ce.item()
            if len(clf_ce_losses) > 0:
                stats['ClfTrain/clf_ce_init'] = clf_ce_losses[0].item()
                if len(clf_ce_losses) > 2:
                    stats['ClfTrain/clf_ce_iter'] = sum(clf_ce_losses[1:-1]).item() / (len(clf_ce_losses) - 2)
        #pdb.set_trace()
        return loss,loss_D, stats


class KYSActor(BaseActor):
    """ Actor for training KYS model """
    def __init__(self, net, objective, loss_weight=None, dimp_jitter_fn=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight

        self.dimp_jitter_fn = dimp_jitter_fn

        # TODO set it somewhere
        self.device = torch.device("cuda:0")

    def __call__(self, data):
        sequence_length = data['test_images'].shape[0]
        num_sequences = data['test_images'].shape[1]

        valid_samples = data['test_valid_image'].to(self.device)
        test_visibility = data['test_visible_ratio'].to(self.device)

        # Initialize loss variables
        clf_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        clf_loss_test_orig_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        dimp_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        test_clf_acc = 0
        dimp_clf_acc = 0

        test_tracked_correct = torch.zeros(num_sequences, sequence_length - 1).long().to(self.device)
        test_seq_all_correct = torch.ones(num_sequences).to(self.device)
        dimp_seq_all_correct = torch.ones(num_sequences).to(self.device)

        is_target_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        is_target_after_prop_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)

        # Initialize target model using the training frames
        train_images = data['train_images'].to(self.device)
        train_anno = data['train_anno'].to(self.device)
        dimp_filters = self.net.train_classifier(train_images, train_anno)

        # Track in the first test frame
        test_image_cur = data['test_images'][0, ...].to(self.device)
        backbone_feat_prev_all = self.net.extract_backbone_features(test_image_cur)
        backbone_feat_prev = backbone_feat_prev_all[self.net.classification_layer]
        backbone_feat_prev = backbone_feat_prev.view(1, num_sequences, -1,
                                                     backbone_feat_prev.shape[-2], backbone_feat_prev.shape[-1])

        if self.net.motion_feat_extractor is not None:
            motion_feat_prev = self.net.motion_feat_extractor(backbone_feat_prev_all).view(1, num_sequences, -1,
                                                                                           backbone_feat_prev.shape[-2],
                                                                                           backbone_feat_prev.shape[-1])
        else:
            motion_feat_prev = backbone_feat_prev

        dimp_scores_prev = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_prev)

        # Remove last row and col (added due to even kernel size in the target model)
        dimp_scores_prev = dimp_scores_prev[:, :, :-1, :-1].contiguous()

        # Set previous frame information
        label_prev = data['test_label'][0:1, ...].to(self.device)
        label_prev = label_prev[:, :, :-1, :-1].contiguous()

        anno_prev = data['test_anno'][0:1, ...].to(self.device)
        state_prev = None

        is_valid_prev = valid_samples[0, :].view(1, -1, 1, 1).byte()

        # Loop over the sequence
        for i in range(1, sequence_length):
            test_image_cur = data['test_images'][i, ...].to(self.device)
            test_label_cur = data['test_label'][i:i+1, ...].to(self.device)
            test_label_cur = test_label_cur[:, :, :-1, :-1].contiguous()

            test_anno_cur = data['test_anno'][i:i + 1, ...].to(self.device)

            # Extract features
            backbone_feat_cur_all = self.net.extract_backbone_features(test_image_cur)
            backbone_feat_cur = backbone_feat_cur_all[self.net.classification_layer]
            backbone_feat_cur = backbone_feat_cur.view(1, num_sequences, -1,
                                                       backbone_feat_cur.shape[-2], backbone_feat_cur.shape[-1])

            if self.net.motion_feat_extractor is not None:
                motion_feat_cur = self.net.motion_feat_extractor(backbone_feat_cur_all).view(1, num_sequences, -1,
                                                                                             backbone_feat_cur.shape[-2],
                                                                                             backbone_feat_cur.shape[-1])
            else:
                motion_feat_cur = backbone_feat_cur

            # Run target model
            dimp_scores_cur = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_cur)
            dimp_scores_cur = dimp_scores_cur[:, :, :-1, :-1].contiguous()

            # Jitter target model output for augmentation
            jitter_info = None
            if self.dimp_jitter_fn is not None:
                dimp_scores_cur = self.dimp_jitter_fn(dimp_scores_cur, test_label_cur.clone())

            # Input target model output along with previous frame information to the predictor
            predictor_input_data = {'input1': motion_feat_prev, 'input2': motion_feat_cur,
                                    'label_prev': label_prev, 'anno_prev': anno_prev,
                                    'dimp_score_prev': dimp_scores_prev, 'dimp_score_cur': dimp_scores_cur,
                                    'state_prev': state_prev,
                                    'jitter_info': jitter_info}

            predictor_output = self.net.predictor(predictor_input_data)

            predicted_resp = predictor_output['response']
            state_prev = predictor_output['state_cur']
            aux_data = predictor_output['auxiliary_outputs']

            is_valid = valid_samples[i, :].view(1, -1, 1, 1).byte()
            uncertain_frame = (test_visibility[i, :].view(1, -1, 1, 1) < 0.75) * (test_visibility[i, :].view(1, -1, 1, 1) > 0.25)

            is_valid = is_valid * ~uncertain_frame

            # Calculate losses
            clf_loss_test_new = self.objective['test_clf'](predicted_resp, test_label_cur,
                                                           test_anno_cur, valid_samples=is_valid)
            clf_loss_test_all[:, i - 1] = clf_loss_test_new.squeeze()

            dimp_loss_test_new = self.objective['dimp_clf'](dimp_scores_cur, test_label_cur,
                                                            test_anno_cur, valid_samples=is_valid)
            dimp_loss_test_all[:, i - 1] = dimp_loss_test_new.squeeze()

            if 'fused_score_orig' in aux_data and 'test_clf_orig' in self.loss_weight.keys():
                aux_data['fused_score_orig'] = aux_data['fused_score_orig'].view(test_label_cur.shape)
                clf_loss_test_orig_new = self.objective['test_clf'](aux_data['fused_score_orig'], test_label_cur, test_anno_cur,  valid_samples=is_valid)
                clf_loss_test_orig_all[:, i - 1] = clf_loss_test_orig_new.squeeze()

            if 'is_target' in aux_data and 'is_target' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_loss_new = self.objective['is_target'](aux_data['is_target'], label_prev, is_valid_prev)
                is_target_loss_all[:, i - 1] = is_target_loss_new

            if 'is_target_after_prop' in aux_data and 'is_target_after_prop' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_after_prop_loss_new = self.objective['is_target'](aux_data['is_target_after_prop'],
                                                                            test_label_cur, is_valid)
                is_target_after_prop_loss_all[:, i - 1] = is_target_after_prop_loss_new

            test_clf_acc_new, test_pred_correct = self.objective['clf_acc'](predicted_resp, test_label_cur, valid_samples=is_valid)
            test_clf_acc += test_clf_acc_new

            test_seq_all_correct = test_seq_all_correct * (test_pred_correct.long() | (1 - is_valid).long()).float()
            test_tracked_correct[:, i - 1] = test_pred_correct

            dimp_clf_acc_new, dimp_pred_correct = self.objective['clf_acc'](dimp_scores_cur, test_label_cur, valid_samples=is_valid)
            dimp_clf_acc += dimp_clf_acc_new

            dimp_seq_all_correct = dimp_seq_all_correct * (dimp_pred_correct.long() | (1 - is_valid).long()).float()

            motion_feat_prev = motion_feat_cur.clone()
            dimp_scores_prev = dimp_scores_cur.clone()
            label_prev = test_label_cur.clone()
            is_valid_prev = is_valid.clone()

        # Compute average loss over the sequence
        clf_loss_test = clf_loss_test_all.mean()
        clf_loss_test_orig = clf_loss_test_orig_all.mean()
        dimp_loss_test = dimp_loss_test_all.mean()
        is_target_loss = is_target_loss_all.mean()
        is_target_after_prop_loss = is_target_after_prop_loss_all.mean()

        test_clf_acc /= (sequence_length - 1)
        dimp_clf_acc /= (sequence_length - 1)
        clf_loss_test_orig /= (sequence_length - 1)

        test_seq_clf_acc = test_seq_all_correct.mean()
        dimp_seq_clf_acc = dimp_seq_all_correct.mean()

        clf_loss_test_w = self.loss_weight['test_clf'] * clf_loss_test
        clf_loss_test_orig_w = self.loss_weight['test_clf_orig'] * clf_loss_test_orig
        dimp_loss_test_w = self.loss_weight.get('dimp_clf', 0.0) * dimp_loss_test

        is_target_loss_w = self.loss_weight.get('is_target', 0.0) * is_target_loss
        is_target_after_prop_loss_w = self.loss_weight.get('is_target_after_prop', 0.0) * is_target_after_prop_loss

        loss = clf_loss_test_w + dimp_loss_test_w + is_target_loss_w + is_target_after_prop_loss_w + clf_loss_test_orig_w

        stats = {'Loss/total': loss.item(),
                 'Loss/test_clf': clf_loss_test_w.item(),
                 'Loss/dimp_clf': dimp_loss_test_w.item(),
                 'Loss/raw/test_clf': clf_loss_test.item(),
                 'Loss/raw/test_clf_orig': clf_loss_test_orig.item(),
                 'Loss/raw/dimp_clf': dimp_loss_test.item(),
                 'Loss/raw/test_clf_acc': test_clf_acc.item(),
                 'Loss/raw/dimp_clf_acc': dimp_clf_acc.item(),
                 'Loss/raw/is_target': is_target_loss.item(),
                 'Loss/raw/is_target_after_prop': is_target_after_prop_loss.item(),
                 'Loss/raw/test_seq_acc': test_seq_clf_acc.item(),
                 'Loss/raw/dimp_seq_acc': dimp_seq_clf_acc.item(),
                 }

        return loss, stats
