# -*- coding: utf-8 -*-
# @Time    : 7/16/21 3:12 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

# the unified ast models for all pretraining/fine-tuning tasks.
'''
AST Model classes
All the functionality/code is the same as the original SSAST, but has been split into two classes to remove branching logic for fine-tuning.

Now ASTModel_pretrain and ASTModel_finetune

Last modified: 07/2023
Author: Daniela Wiepert, source
Email: wiepert.daniela@mayo.edu
File: ast_models.py
'''
#IMPORTS
#built-in
import random
from random import randrange

#third party
import numpy as np
import timm
import torch.nn as nn
import torch

from timm.models.layers import trunc_normal_, to_2tuple

#local
from ._classification_heads import BasicClassifier

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ASTModel_pretrain(nn.Module):
    '''
    Edited pretraining class
    '''
    def __init__(self, fshape=128, tshape=2, fstride=128, tstride=2,
                 input_fdim=128, input_tdim=1024, model_size='base', checkpoint=None):
        super(ASTModel_pretrain, self).__init__()
        #assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code is NOT compatible with newer versions.'
            # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        #TODO: fix
        if checkpoint != None:
            print('loading pretrained model not yet supported')
            raise ValueError('checkpoint should be None')
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     sd = torch.load(checkpoint, map_location=device)
        #     # get the fshape and tshape, input_fdim and input_tdim in the pretraining stage
        #     try:
        #         p_fshape, p_tshape = sd['module.v.patch_embed.proj.weight'].shape[2], sd['module.v.patch_embed.proj.weight'].shape[3]
        #         p_input_fdim, p_input_tdim = sd['module.p_input_fdim'].item(), sd['module.p_input_tdim'].item()
        #     except:
        #         raise  ValueError('The model loaded is not from a torch.nn.Dataparallel object. Wrap it with torch.nn.Dataparallel and try again.')
        #         #raise ValueError('Setting checkpoint at pretraining stage is useless, pretraining is always from scratch, please change it to None.')
        if fstride != fshape or tstride != tshape:
            raise ValueError('fstride != fshape or tstride != tshape, they must be same at the pretraining stage, patch split overlapping is not supported.')

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
                # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
       
        pretrained=False

        if model_size == 'tiny':
            self.v = timm.create_model('deit_tiny_distilled_patch16_224', img_size=(input_fdim, input_tdim),pretrained=pretrained)
            #self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=pretrained)
            self.heads, self.depth = 3, 12
            self.cls_token_num = 2
        elif model_size == 'small':
            self.v = timm.create_model('deit_small_distilled_patch16_224',img_size=(input_fdim, input_tdim), pretrained=pretrained)
            #self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=pretrained)
            self.heads, self.depth = 6, 12
            self.cls_token_num = 2
        elif model_size == 'base':
            #img = Image.open(urlopen("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"))
            #self.v = timm.create_model('deit_base_distilled_patch16_384.fb_in1k', pretrained=pretrained)
            self.v = timm.create_model('deit_base_distilled_patch16_384', img_size=(input_fdim, input_tdim), pretrained=pretrained)
            #self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=pretrained)
            self.heads, self.depth = 12, 12
            self.cls_token_num = 2
        elif model_size == 'base_nokd':
            self.v = timm.create_model('deit_based_patch16_384',img_size=(input_fdim, input_tdim), pretrained=pretrained)
            #self.v = timm.create_model('vit_deit_base_patch16_384', pretrained=pretrained)
            self.heads, self.depth = 12, 12
            self.cls_token_num = 1
        else:
            raise Exception('Model size must be one of tiny, small, base, base_nokd')

        #timm.models.vision_transformer.PatchEmbed = PatchEmbed

        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]

        # SSL Pretraining Code
        self.softmax = nn.Softmax(dim=-1)
        self.lsoftmax = nn.LogSoftmax(dim=-1)
        self.fshape, self.tshape = fshape, tshape
        self.fstride, self.tstride = fstride, tstride
        self.input_fdim, self.input_tdim = input_fdim, input_tdim
        # this is a trick to make state_dict to track pretraining input_fdim and input_tdim and save them by using torch.save
        self.p_input_fdim, self.p_input_tdim = nn.Parameter(torch.tensor(input_fdim), requires_grad=False), nn.Parameter(torch.tensor(input_tdim), requires_grad=False)

        # masked patch classification (discriminative objective) layer
        # we use two layers for pretext task, but using a single layer has similar performance.
        # we map the output of transformer (768-dim for base models) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
        # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
        self.cpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
        # masked patch reconstruction (generative objective) layer
        self.gpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
        self.unfold = torch.nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))

        # we use learnable mask embedding (follow the BEIT paper), but using a fixed mask embedding (e.g., 0) leads to same performance.
        self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
        self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

        # get the intermediate shape
        self.p_f_dim, self.p_t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
        num_patches = self.p_f_dim * self.p_t_dim
        self.num_patches = num_patches
        self.v.patch_embed.num_patches = num_patches
        print('pretraining patch split stride: frequency={:d}, time={:d}'.format(fstride, tstride))
        print('pretraining patch shape: frequency={:d}, time={:d}'.format(fshape, tshape))
        print('pretraining patch array dimension: frequency={:d}, time={:d}'.format(self.p_f_dim, self.p_t_dim))
        print('pretraining number of patches={:d}'.format(num_patches))

        # the linear patch projection layer, use 1 channel for spectrogram rather than the original 3 channels for RGB images.
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        self.v.patch_embed.proj = new_proj

        # use trainable positional embedding
        new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + self.cls_token_num, self.original_embedding_dim))
        self.v.pos_embed = new_pos_embed
        trunc_normal_(self.v.pos_embed, std=.02)


        #TODO: fix
        # if checkpoint != None:
        #     self.v = torch.nn.DataParallel(self.v)
        #     self.v.load_state_dict(sd, strict=False)
        #     self.v = self.v.module.v
    
# get the shape of intermediate representation.
    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    # generate mask for 16*16 patch
    def gen_maskid_patch(self, sequence_len=512, mask_size=100, cluster=3): 
        mask_id = []

        # randomize clutering factor in [3,6)
        cur_clus = randrange(cluster) + 3

        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)

            # this improves the efficiency, but might change the pretrained model
            # while start_id in mask_id:
            #     start_id = randrange(sequence_len)

            cur_mask = []
            for i in range(0, cur_clus):
                for j in range(0, cur_clus):
                    mask_cand = start_id + self.p_t_dim * i + j
                    if mask_cand > 0 and mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)

    # using cluster for frame masking hurts the performance, so just use the naive random sampling
    def gen_maskid_frame(self, sequence_len=512, mask_size=100): 
        mask_id = random.sample(range(0, sequence_len), mask_size)
        return torch.tensor(mask_id)
    
     # masked patch pretraining with discriminative objective
    def mpc(self, x, mask_patch, cluster, show_mask=False):
        input = self.unfold(x).transpose(1, 2) # (batch size, 512, 256)
        B = x.shape[0]
        # x in shape (batch_size, sequence_len, embedding dim)
        x = self.v.patch_embed(x)

        # encode the patch
        # size 12(batch_size) * 100(#mask_patch) * 768(hidden_dim), prepare to save the true values of masked samples
        encode_samples = torch.empty((B, mask_patch, 256), device=x.device, requires_grad=False).float()
        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)

        # for each audio clip in the batch
        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            if cluster == True:
                # use this if you are masking e.g. 16*16 patches
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                # use this if you are masking frame, i.e., 128*2 patches
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            # copy the masked embeddings, note gradients are stopped in this path
            encode_samples[i] = input[i, mask_index[i], :].clone().detach()
            # mask the encode samples with 0
            mask_dense[i, mask_index[i], :] = 0

        # follow BEIT paper, mask with learnable masking embedding, but no performance diff observed compared with masking with 0s.
        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # mask the patch
        x = x * mask_dense + (1-mask_dense) * mask_tokens

        # pass through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        # prediction of the masked patch
        pred = torch.empty((B, mask_patch, 256), device=x.device).float()  # e.g. size 12*100*768
        for i in range(B):
            #  +2 for indexes because skipping the cls and dis token
            # we map the output of transformer (768-dim for base models) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
            # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
            pred[i] = self.cpredlayer(x[i, mask_index[i] + self.cls_token_num, :])

        # calculate the NCE loss
        nce = torch.tensor(0.0).to(x.device)
        correct = torch.tensor(0.0).to(x.device)
        for i in np.arange(0, B):
            # negative samples are from the same batch
            # equation (1) of the ssast paper
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 100*100
            correct += torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, mask_patch, device=x.device)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        acc = 1. * correct / (B * mask_patch)
        nce = nce / (-1. * B * mask_patch)

        # visualize the masked area, for probing test only, set show_mask = False for any training/inference.
        if show_mask == False:
            return acc, nce
        else:
            if B > 1:
                raise Exception('Currently only support single spectrogram probing test.')

            self.mask_correct = torch.nn.Parameter(torch.arange(0, mask_patch), requires_grad=False)

            pred = input.clone()  # [B, 512, 256]
            masked = input.clone()

            for i in range(B):
                result = [float(t) * 99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)]
                pred[i, mask_index[i], :] = torch.tensor(result).reshape(mask_patch, 1).expand(mask_patch, 256)
                masked[i, mask_index[i], :] = 99.0

            # print(total)
            # print(self.softmax(total))
            # print(torch.argmax(self.softmax(total), dim=0))
            # print(self.mask_correct)
            # print(torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct))
            # print([float(t)*99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)])

            fold = torch.nn.Fold(output_size=([self.input_fdim, self.input_tdim]), kernel_size=(self.fshape, self.tshape), stride=(self.fstride, self.tstride))
            pred = fold(pred.transpose(1, 2))
            masked = fold(masked.transpose(1, 2))

            return pred, masked

    # # masked patch pretraining with generative objective
    def mpg(self, input, mask_patch, cluster):
        B = input.shape[0]
        x = self.v.patch_embed(input)
        input = self.unfold(input).transpose(1, 2)

        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            if cluster == True:
                # use this if you are masking e.g. 16*16 patches
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                # use this if you are masking frame, i.e., 128*2 patches
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            mask_dense[i, mask_index[i], :] = 0

        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # follow BEIT paper, mask with learnable masking embedding, but no performance diff observed compared with masking with 0s.
        x = x * mask_dense + (1-mask_dense) * mask_tokens

        # go through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        pred = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float()  # e.g. size 12*100*256
        target = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float() # e.g. size 12*100*256

        for i in range(B):
            #  +2 for indexes because cls and dis token
            pred[i] = self.gpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
            target[i] = input[i, mask_index[i], :]

        # calculate the MSE loss
        mse = torch.mean((pred - target) ** 2)

        return mse
    
    def forward(self, x, task, cluster=True, mask_patch=100):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1) #(batch_size, 1, time_frame_num, frequency_bins)
        x = x.transpose(2, 3) #(batch_size, 1, frequency_bins, time_frame_num)

      # pretraining, masked patch classification (discriminative objective)
        if task == 'pretrain_mpc':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster)
        # pretraining, masked patch reconstruction (generative objective)
        elif task == 'pretrain_mpg':
            return self.mpg(x, mask_patch=mask_patch, cluster=cluster)
        elif task == 'visualize_mask':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster, show_mask=True)
        else:
            raise Exception('Task unrecognized.')

class ASTModel_finetune(nn.Module):
    '''
    Edited finetuning class
    Parameters:
    :param task: finetuning task
    :param label_dim: default
    :param fshape: default
    :param tshape: default
    :param fstride: default
    :param tstride: default
    :param input_fdim: # of frequency bins
    :param input_tdim: # of time frames
    :param model_size: model size to initialize - will run into errors if pretrained model path is a mismatch with this
    :param checkpoint: path to a pretrained model checkpoint
    :param freeze: specify whether to freeze the pretrained model parameters
    :param weighted: specify which mode to run as the forward function (False: forward for a single hidden layer, True: weighted layer sum)
    :param layer: layer for single hidden layer extraction, default is -1 (which extracts the final hidden layer)
    :param shared_dense: specify whether to add a shared dense layer before classification head
    :param sd_bottleneck: size to reduce to in shared dense layer
    :param activation: activation function for classification head
    :param final_dropout: amount of dropout to use in classification head
    :param layernorm: include layer normalization in classification head
    :param clf_bottleneck: size to reduce to in intial classifier dense layer
    '''
    def __init__(self, task='ft_cls',label_dim=527,
                 fshape=128, tshape=2, fstride=128, tstride=2,
                 input_fdim=128, input_tdim=1024, model_size='base',checkpoint=None,
                 freeze=True, weighted=False, layer=-1, shared_dense=False, sd_bottleneck=768,
                 activation='relu', final_dropout=0.2, layernorm=True, clf_bottleneck=768):

        ######### ORIGINAL CODE  ######### 
        super(ASTModel_finetune, self).__init__()
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if checkpoint == None:
            raise ValueError('Please set checkpoint to load a pretrained models.')
        sd = torch.load(checkpoint, map_location=device)

        # if not isinstance(sd, torch.nn.DataParallel):
        #     sd = torch.nn.DataParallel(sd)
        # get the fshape and tshape, input_fdim and input_tdim in the pretraining stage
        #need to conf
        try:
            p_fshape, p_tshape = sd['module.v.patch_embed.proj.weight'].shape[2], sd['module.v.patch_embed.proj.weight'].shape[3]
            p_input_fdim, p_input_tdim = sd['module.p_input_fdim'].item(), sd['module.p_input_tdim'].item()
        except:
            raise  ValueError('The model loaded is not from a torch.nn.Dataparallel object. Wrap it with torch.nn.Dataparallel and try again.')

        print('now load a SSL pretrained models from ' + str(checkpoint))
        # during pretraining, fstride=fshape and tstride=tshape because no patch overlapping is used
        # here, input_fdim and input_tdim should be that used in pretraining, not that in the fine-tuning.
        # we need to know input_fdim and input_tdim to do positional embedding cut/interpolation.
        # generally it should be better to use same input_fdim during pretraining and finetuning, but input_tdim can be safely different
        audio_model = ASTModel_pretrain(fstride=p_fshape, tstride=p_tshape, fshape=p_fshape, tshape=p_tshape,
                                input_fdim=p_input_fdim, input_tdim=p_input_tdim, model_size=model_size)
        audio_model = torch.nn.DataParallel(audio_model)
        audio_model.load_state_dict(sd, strict=False)

        self.v = audio_model.module.v
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.cls_token_num = audio_model.module.cls_token_num


        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
        # patch array dimension during pretraining
        p_f_dim, p_t_dim = audio_model.module.p_f_dim, audio_model.module.p_t_dim
        num_patches = f_dim * t_dim
        p_num_patches = p_f_dim * p_t_dim
        self.v.patch_embed.num_patches = num_patches
        print('fine-tuning patch split stride: frequncey={:d}, time={:d}'.format(fstride, tstride))
        print('fine-tuning number of patches={:d}'.format(num_patches))

        # patch shape should be same for pretraining and fine-tuning
        if fshape != p_fshape or tshape != p_tshape:
            raise ValueError('The patch shape of pretraining and fine-tuning is not consistant, pretraining: f={:d}, t={:d}, finetuning: f={:d}, t={:d}'.format(p_fshape, p_tshape, fshape, tshape))

        # patch split stride generally should be different for pretraining and fine-tuning, as patch split overlapping is only used in finetuning
        # during pretraining, p_fshape = p_fstride and p_tshape = p_tstride
        if fstride != p_fshape or tstride != p_tshape:
            # initialize a new patch embedding layer with desired new stride.
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
            # but the weights of patch embedding layer is still got from the pretrained models
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

        new_pos_embed = self.v.pos_embed[:, self.cls_token_num:, :].detach().reshape(1, p_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, p_f_dim, p_t_dim)
        # cut or interpolate the positional embedding
        if t_dim < p_t_dim:
            new_pos_embed = new_pos_embed[:, :, :, int(p_t_dim/2) - int(t_dim / 2): int(p_t_dim/2) - int(t_dim / 2) + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(8, t_dim), mode='bilinear')
        if f_dim < p_f_dim:
            new_pos_embed = new_pos_embed[:, :, int(p_f_dim/2) - int(f_dim / 2): int(p_f_dim/2) - int(f_dim / 2) + t_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

        new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
        self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :self.cls_token_num, :].detach(), new_pos_embed], dim=1))

        ############ ADDITIONS ############
        self.sd_bottleneck=sd_bottleneck
        self.clf_bottleneck=clf_bottleneck

        self.shared_dense = shared_dense
        if self.shared_dense:
            self.dense = nn.Linear(self.original_embedding_dim, self.sd_bottleneck)
            self.clf_input = self.sd_bottleneck
        else:
            self.clf_input = self.original_embedding_dim

        self.weighted = weighted #specification for running weight sum finetuning, where we generate a weight for each layer to contribute to the classification

        self.n_states = len(self.v.blocks) + 1
        if self.weighted:
            self.weightsum=nn.Parameter(torch.ones(self.n_states)/self.n_states) ## this is hard coded for DeiT currently, but could be changed for other models
        else:
            self.weightsum=torch.ones(self.n_states)/self.n_states #non-parameter version

        #specify which hidden layer is being used for input to classification. 
        assert layer >= -1 and layer <= self.n_states, f'invalid layer given: {layer}. Layer must either be -1 for final layer, or a number between 0 and {self.n_states}'
        self.layer = layer
        
        # mlp head for fine-tuning
        self.mlp_heads = []
        self.label_dim = label_dim
        if isinstance(self.label_dim, list):
            for dim in self.label_dim:
                self.mlp_heads.append(BasicClassifier(input_size=self.clf_input, bottleneck=self.clf_bottleneck, output_size=dim,
                                             activation=activation, final_dropout=final_dropout,layernorm=layernorm))
        
        else:
            self.mlp_heads.append(BasicClassifier(input_size=self.clf_input, bottleneck=self.clf_bottleneck, output_size=self.label_dim,
                                             activation=activation, final_dropout=final_dropout,layernorm=layernorm))

        self.mlp_heads = nn.ModuleList(self.mlp_heads)   
        
        # if you don't want to finetune the entire model, but only the classification head, all parameters in the base model (self.v) are frozen
        if freeze:
            for param in self.v.parameters():
                param.requires_grad = False

        #set up merging function based on finetuning task
        self.task = task
        if self.task == 'ft_cls':
            self._merge_fn = self._cls
        elif self.task == 'ft_avgtok':
            self._merge_fn = self._avgtok
        else:
            raise ValueError(f'Task is set as {task}, but this is an invalid task. Please set finetuning task as either ft_cls or ft_avgtok')

    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        """
        Original fn to get shape of intermediate representation
        """
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    
    def _base_model(self, x):
        """
        Split from original finetuning functions. Runs input through the base model.

        Code added to get hidden states (output of each transformer block). 
        Output changed to list of hidden states, with index 0 being the prepared input that goes into the first transformer block
        indices [1-12] being the outputs after each block, and 13(-1) being the output of the final block wrapped with a normalization layer.
        :param x: fbank input (batch size, freq bins, time frames)
        :return hidden_states: list of hidden ssast states, list is len 14, each item is a tensor of size (batch size, hidden size, embedding dim)
        """
        B = x.shape[0]
        x = self.v.patch_embed(x) #(batch size x patch_embed out x hidden_dim/embedding_dim) (1, 512, 768)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1) # batch_size, 1, embedding dim
            dist_token = self.v.dist_token.expand(B, -1, -1) #batch_size, 1, embedding dim
            x = torch.cat((cls_tokens, dist_token, x), dim=1) #add both at top (batch size x patch_embed out + 2 x embedding dim) (1, 514, 768), 
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1) #add at top, (1, 513, 768)
        x = x + self.v.pos_embed #no shape change
        x = self.v.pos_drop(x) #no shape change

        hidden_states=[]
        hidden_states.append(x) 

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
            hidden_states.append(x) #get each hidden state
        x = self.v.norm(x)
        hidden_states.append(x) #NOTE that the final two hidden states are the same block output, with one being directly from the model and one being the normalized

        return hidden_states #14 hidden states, each of size (batch size, patch_embed out + # tokens, embedding_dim) 
    
    def _mat_mul_weights(self, hidden_states):
        """
        Private function that takes in hidden states that have been merged (using _cls or _avgtok), and multiplies it by the weighted sum
        :param hidden_states: torch tensor of hidden ssast states (# hidden layer, batch size, embedding dim)
        :return: matrix multiplication of hidden states and weighted sum parameter (batch size, embedding dim)
        """
        hidden = torch.permute(hidden_states, (1,0,2)) #permute the shape so it is batch size x # hidden layers x embedding dim
        w_sum = torch.reshape(self.weightsum,(1,1,self.n_states)) #reshape weight sum to have the right number of dims
        w_sum=w_sum.repeat(hidden.shape[0],1,1) #repeat for each sample in batch to allow for batch matrix product, results in size #batch size x 1 x 13
        weighted_sum=torch.bmm(w_sum, hidden) #output: batch size x 1 x embedding dim

        return torch.squeeze(weighted_sum, dim=1) #need to squeeze at output going into classifier should be batch size x embedding dim 

    def _cls(self, hidden_states, weighted, layer):
        """
        CLS token mean merging strategy for finetuning.
        Based on the original function but altered to be compatible with weighted sum (which has an additional dimension)
        and to extract a specific hidden layer 
        :param hidden_states: list of hidden ssast states, list is len 14, each item is a tensor of size (batch size, hidden size, embedding dim)
        :param weighted: boolean specifying whether running weighted sum 
        :param layer: int indicating which hidden state layer to use.
        :return outputs: hidden state(s) merged to be of size (batch size, embedding dim)
        """
        if weighted:
            x = torch.stack(hidden_states[:-1]) #stack all hidden states (EXCLUDING THE LAST ITEM which is just the output of the last transformer block after being run through a normalization layer. 2nd to last ind is same output prior to norm layer)
            x = x[:, :, :self.cls_token_num,:]
            x = torch.mean(x, dim=2) #hidden layers x batch size x embedding dim
            outputs = self._mat_mul_weights(x) #run matrix multiplication
        else:  
            x = hidden_states[layer] #select which output you are using as x
            x = x[:,:self.cls_token_num,:]
            outputs = torch.mean(x,dim=1)
        
        return outputs

    def _avgtok(self, hidden_states, weighted, layer):
        """
        Mean merging strategy for finetuning.
        Based on the original function but altered to be compatible with weighted sum (which has an additional dimension)
        and to extract a specific hidden layer 
        :param hidden_states: list of hidden ssast states, list is len 14, each item is a tensor of size (batch size, hidden size, embedding dim)
        :param weighted: boolean specifying whether running weighted sum 
        :param layer: int indicating which hidden state layer to use.
        :return outputs: hidden state(s) merged to be of size (batch size, embedding dim)
        """
        if weighted: 
            x = torch.stack(hidden_states[:-1])  #stack all hidden states (EXCLUDING THE LAST ITEM which is just the output of the last transformer block after being run through a normalization layer. 2nd to last ind is same output prior to norm layer)
            x = x[:,:,self.cls_token_num:,:]  #select all EXCEPT the cls token
            x = torch.mean(x, dim=2) #hidden layers x batch size x embedding dim
            outputs = self._mat_mul_weights(x)
        else:
            x = hidden_states[layer] #select which output you are using as x
            x = x[:,self.cls_token_num:,:] #select all output except the tokens (exclud the first 1 or two of Dim 1)
            outputs = torch.mean(x, dim=1)
        
        return outputs

    def extract_embedding(self, x, embedding_type='ft', layer=None, task=None, pooling_mode="mean"):
        """
        Extract an embedding from various parts of the model
        :param x: fbank input (batch size, freq bins, time frames)
        :param embedding_type: 'ft', 'pt', or 'wt', to indicate whether to extract from classification head (ft), hidden state (pt), or weighted sum mat mul (wt)
        :param layer: int indicating which hidden state layer to use.
        :param task: finetuning task, only used for 'pt' or 'wt' embedding extraction.
        :param pooling_mode: method of pooling embeddings if required ("mean" or "sum")
        :return e: embeddings for a batch (batch_size, embedding dim)
        """
        ## EMBEDDING 'ft': extract from finetuned classification head
        if embedding_type == 'ft':
            assert pooling_mode == 'mean' or pooling_mode == 'sum', f"Incompatible pooling given: {pooling_mode}. Please give mean or sum"

            #register a forward hook to grab the output of the first classification layer (called 'dense')
            activation = {}
            def _get_activation(name):
                def _hook(model, input, output):
                    activation[name] = output.detach()
                return _hook
            
            x = x.unsqueeze(1) #(batch_size, 1, time_fram_num, frequency_bins), e.g. (12, 1, 1024, 128)
            x = x.transpose(2, 3) #(batch_size, 1, frequency_bines, time_frame_num) e.g. (12, 1, 128, 1024)

            hidden_states = self._base_model(x)
            if layer is None:
                layer = self.layer

            x= self._merge_fn(hidden_states, self.weighted, self.layer)
            
            if self.shared_dense:
                x = self.dense(x)
            
            embeddings = []
            for clf in self.mlp_heads:
                clf.head.dense.register_forward_hook(_get_activation('embeddings')) 
                logits = clf(x)
                embeddings.append(activation['embeddings'])
            
            embeddings = torch.stack(embeddings, dim=1)
            if pooling_mode == "mean":
                e = torch.mean(embeddings, dim=1)
            else:
                e = torch.sum(embeddings, dim=1)
        
        ## EMBEDDING 'pt': extract from a hidden state, 'wt': extract after matmul with layer weights
        elif embedding_type in ['pt', 'st', 'wt']:
            #dealing with weighted sum
            #if embedding type is st, do we want it to follow whatever the base model did?
            if embedding_type == 'pt': #if the embedding is extracting from pre-trained model, weighted must be false
                weighted=False
            elif embedding_type == 'wt': #else the model must have been trained for weighted sum, so original self.weighted must be True
                try:
                    weighted=self.weighted
                    assert weighted
                except:
                    raise ValueError('The model must be trained for weightsum')
            else:
                #if embedding type is st, it will follow whatever you desire
                weighted=self.weighted

            #dealing with specific layer extraction
            #if pretrained extraction, you can extract from any layer, and if none specified, use the one from the model
            #if weighted sum, embedding extraction should use all the same layer as the model
            #if shared dense, embedding extraction should use the same layer as the model
            if embedding_type in ['st','wt'] or layer is None:
               layer = self.layer

            x = x.unsqueeze(1) #(batch_size, 1, time_frame_num, frequency_bins), e.g. (12, 1, 1024, 128)
            x = x.transpose(2, 3) #(batch_size, 1, frequency_bines, time_frame_num) e.g. (12, 1, 128, 1024)

            hidden_states = self._base_model(x)

            if task is None: 
                e = self._merge_fn(hidden_states, weighted, layer)
            elif task == 'ft_cls':
                e = self._cls(hidden_states, weighted, layer)
            elif task == 'ft_avgtok':
                e = self._avgtok(hidden_states, weighted, layer)
            else:
                raise ValueError('Selected task not valid')
            
            if embedding_type == 'st':
                assert self.shared_dense == True, 'The model must be trained with a shared dense' 
                e = self.dense(e)
            
        else:
            raise ValueError('Embedding type must be finetune (ft), pretrain (pt), shared dense (st), or weighted sum (wt)')
        
        return e
    
    def forward(self, x):
        """
        Forward function for the model
        Reshapes input, runs through base model, merges the hidden states, runs through the classification head
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1) #(batch_size, 1, time_fram_num, frequency_bins), e.g. (12, 1, 1024, 128)
        x = x.transpose(2, 3) #(batch_size, 1, frequency_bines, time_frame_num) e.g. (12, 1, 128, 1024)

        hidden_states = self._base_model(x)
        x = self._merge_fn(hidden_states, self.weighted, self.layer)
        
        if self.shared_dense:
            x = self.dense(x)

        preds = []
        for clf in self.mlp_heads:
            pred = clf(x)
            preds.append(pred)

        logits = torch.column_stack(preds)

        return logits
    

