import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MSELoss, BCEWithLogitsLoss


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation="ReLU", kernel_size=3, dilation=1, padding_mode='circular'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, dilation=dilation, padding='same', padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, dilation=dilation, padding='same', padding_mode=padding_mode)

        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "elu":
            self.activation = nn.ELU()
        elif activation.lower() == "leakyrelu":
            self.activation = nn.LeakyReLU()

    def forward(self, x):
        f = self.activation(self.conv1(x))
        f = self.activation(self.conv2(f))

        return f


class PreResBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, activation="ReLU", kernel_size=3, dilation=1, padding_mode='circular'):
        super().__init__()
        
        if out_ch is None:
            out_ch = in_ch
            self.skipconv = nn.Identity()
        else:
            self.skipconv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "elu":
            self.activation = nn.ELU()
        elif activation.lower() == "leakyrelu":
            self.activation = nn.LeakyReLU()

        med_ch = in_ch//2 if in_ch > 1 else 1

        self.bn1 = nn.BatchNorm2d(num_features=in_ch)
        self.bn2 = nn.BatchNorm2d(num_features=med_ch)
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=med_ch, kernel_size=kernel_size, dilation=dilation, padding='same', padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(in_channels=med_ch, out_channels=med_ch, kernel_size=kernel_size, dilation=dilation, padding='same', padding_mode=padding_mode)
        self.outconv = nn.Conv2d(in_channels=med_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        f = self.conv1(self.activation(self.bn1(x)))
        f = self.conv2(self.activation(self.bn2(f)))
        f = self.outconv(f)

        return f + self.skipconv(x)


class ScaleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation="ReLU", scale_depth=1, kernel_size=3, dilation=1, padding_mode='circular'):
        super().__init__()
        
        self.first_preres_blocks = nn.ModuleList( [ PreResBlock(in_ch, None, activation, kernel_size, dilation, padding_mode=padding_mode) for _ in range(scale_depth-1) ] )
        self.last_preres_block = PreResBlock(in_ch, out_ch, activation, kernel_size, dilation, padding_mode=padding_mode)

    def forward(self, x):
        f = x
        if len(self.first_preres_blocks) > 0:
            for block in self.first_preres_blocks:
                f = block(f)
        f = self.last_preres_block(f)

        return f


class SimpleEncoder(nn.Module):
    def __init__(self, hlist, num_latent_channels, s, activation="ReLU"):
        super().__init__()
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(2,num_latent_channels,s,s))
        self.encoder_layers = nn.ModuleList( [ nn.Linear(hlist[i], hlist[i+1]) for i in range(len(hlist)-1) ] )
        
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "elu":
            self.activation = nn.ELU()
        elif activation.lower() == "leakyrelu":
            self.activation = nn.LeakyReLU()

    def forward(self, x):
        f = self.flatten(x)
        for layer in self.encoder_layers:
            f = layer(f)
            f = self.activation(f)
        f = self.unflatten(f)
        
        mean, logstd = f[:,0,:], f[:,1,:]
        return mean, logstd


class Encoder(nn.Module):
    def __init__(self, chs, activation="ReLU", scale_depth=1, kernel_size=None, dilation=None, padding_mode='circular'):
        super().__init__()
        self.downsampling = nn.AvgPool2d(kernel_size=2)
        self.encoder_blocks = nn.ModuleList( [ ScaleBlock(chs[i], chs[i+1], activation, scale_depth[i+1], kernel_size[i+1], dilation[i+1], padding_mode=padding_mode) for i in range(len(chs)-1) ] )

    def forward(self, x):
        encoder_feature_maps = [x]
        f = x
        for block in self.encoder_blocks:
            f = self.downsampling(f)
            f = block(f)
            encoder_feature_maps.append(f)

        encoder_feature_maps.reverse()
        return encoder_feature_maps


class Decoder(nn.Module):
    def __init__(self, chs, latent_num=0, activation="ReLU", scale_depth=1, kernel_size=None, dilation=None, padding_mode='circular', latent_channels=None, latent_locks=None):
        super().__init__()
        self.depth = len(chs) - 1
        self.latent_num = latent_num
        self.latent_channels = latent_channels
        self.latent_locks = latent_locks

        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)

        # Prior Net
        self.latent_mean_convs = nn.ModuleList(
            [ nn.Conv2d(in_channels=chs[i], out_channels=latent_channels[i], kernel_size=1) for i in range(latent_num) ] 
        )

        self.latent_std_convs = nn.ModuleList(
            [ nn.Conv2d(in_channels=chs[i], out_channels=latent_channels[i], kernel_size=1) for i in range(latent_num) ]
        )
        self.decoder_blocks = nn.ModuleList(
            [ ScaleBlock(chs[i] + chs[i+1] + latent_channels[i], chs[i+1], activation, scale_depth[i+1], kernel_size[i+1], dilation[i+1], padding_mode=padding_mode) if i < latent_num
            else ScaleBlock(chs[i] + chs[i+1], chs[i+1], activation, scale_depth[i+1], kernel_size[i+1], dilation[i+1], padding_mode=padding_mode) for i in range(self.depth) ]
        )

        # Posterior Net
        self.post_latent_mean_convs = nn.ModuleList(
            [ nn.Conv2d(in_channels=chs[i], out_channels=latent_channels[i], kernel_size=1) for i in range(latent_num) ]
        )
        self.post_latent_std_convs = nn.ModuleList(
            [ nn.Conv2d(in_channels=chs[i], out_channels=latent_channels[i], kernel_size=1) for i in range(latent_num) ]
        )
        self.post_decoder_blocks = nn.ModuleList(
            [ ScaleBlock(chs[i] + chs[i+1] + latent_channels[i], chs[i+1], activation, scale_depth[i+1], kernel_size[i+1], dilation[i+1], padding_mode=padding_mode) for i in range(latent_num - 1) ]
        )


    def sample_latent(self, means, log_stds, latent_lock):
        if latent_lock is True:
            latent = means
        else:
            rands = torch.normal( mean=torch.zeros_like(means), std=torch.ones_like(log_stds) )
            latent = rands * torch.exp(log_stds) + means

        return latent
    

    def forward(self, feature_maps, post_feature_maps=None, insert_from_postnet=False):
        if post_feature_maps is None:  # Not Using Posterior Net
            f = feature_maps[0]
            for i in range(self.depth):
                if i < self.latent_num:
                    means, log_stds = self.latent_mean_convs[i](f), self.latent_std_convs[i](f)
                    latent = self.sample_latent(means, log_stds, self.latent_locks[i])

                    f = torch.cat([f, latent], dim=1)

                f = self.upsampling(f)

                f = torch.cat([ f, feature_maps[i+1] ], dim=1)
                f = self.decoder_blocks[i](f)

            if self.latent_num == self.depth + 1:
                means, log_stds = self.latent_mean_convs[self.depth](f), self.latent_std_convs[self.depth](f)
                latent = self.sample_latent(means, log_stds, self.latent_locks[self.depth])

                f = torch.cat([f, latent], dim=1)


            # Items to return as well as the network's output
            # infodict = {
            #     'prior_latents': prior_latents,
            #     'prior_means': prior_means,
            #     'prior_stds': prior_stds
            # }

            return f
        
        
        else:  # Using Posterior Net
            post_means, post_stds = [], []
            post_latents = []
            l = post_feature_maps[0]
            for i in range(self.latent_num):
                means, log_stds = self.post_latent_mean_convs[i](l), self.post_latent_std_convs[i](l)
                post_means.append(means)
                post_stds.append(torch.exp(log_stds))
                post_latent = self.sample_latent(means, log_stds, self.latent_locks[i])

                post_latents.append(post_latent)

                if i < self.latent_num - 1:
                    l = torch.cat([l, post_latent], dim=1)
                    l = self.upsampling(l)

                    l = torch.cat([ l, post_feature_maps[i+1] ], dim=1)
                    l = self.post_decoder_blocks[i](l)

            prior_means, prior_stds = [], []
            prior_latents = []
            f = feature_maps[0]
            for i in range(self.depth):
                if i < self.latent_num:
                    means, log_stds = self.latent_mean_convs[i](f), self.latent_std_convs[i](f)
                    prior_means.append(means)
                    prior_stds.append(torch.exp(log_stds))
                    
                    if (self.training is True and self.latent_locks[i] is False) or (insert_from_postnet is True):  # Insert Latents from Posterior Net
                        latent = post_latents[i]
                    else:  # Insert Latents from Prior Net
                        latent = self.sample_latent(means, log_stds, self.latent_locks[i])

                    prior_latents.append(latent)

                    f = torch.cat([f, latent], dim=1)

                f = self.upsampling(f)

                f = torch.cat([ f, feature_maps[i+1] ], dim=1)
                f = self.decoder_blocks[i](f)
            
            if self.latent_num == self.depth + 1:
                means, log_stds = self.latent_mean_convs[self.depth](f), self.latent_std_convs[self.depth](f)
                prior_means.append(means)
                prior_stds.append(torch.exp(log_stds))
                
                if (self.training is True and self.latent_locks[self.depth] is False) or (insert_from_postnet is True):  # Insert Latents from Posterior Net
                    latent = post_latents[self.depth]
                else:  # Insert Latents from Prior Net
                    latent = self.sample_latent(means, log_stds, self.latent_locks[self.depth])

                prior_latents.append(latent)

                f = torch.cat([f, latent], dim=1)


            # Calculate kl divergence between posterior net and prior net latents
            kls = torch.zeros(self.latent_num, device=f.device) if self.latent_num > 0 else None  # next(self.parameters()).device

            for i in range(self.latent_num):
                if self.latent_locks[i] is False:
                    kl = torch.log( prior_stds[i] / post_stds[i] )                             \
                        + (    post_stds[i] ** 2  +  (post_means[i] - prior_means[i]) ** 2)     \
                            / 2 / prior_stds[i] ** 2                                         \
                        - 1 / 2
                    kls[i] = kl.sum(dim=(1,2,3)).mean()

            
            # Items to return as well as the network's output
            infodict = {
                'kls': kls,
                'post_latents': post_latents,
                'prior_latents': prior_latents,
                'post_means': post_means,
                'post_stds': post_stds,
                'prior_means': prior_means,
                'prior_stds': prior_stds
            }
       
            return f, infodict


class HPUNet(nn.Module):
    def __init__(self, in_ch, chs, latent_num=0, out_ch=1, activation="ReLU", scale_depth=None, kernel_size=None, dilation=None, padding_mode='circular', latent_channels=None, latent_locks=None):
        super().__init__()
        if latent_locks is None:
            latent_locks = [False for _ in range(latent_num)]
        if latent_channels is None:
            latent_channels = [1 for _ in range(latent_num)]
        assert len(latent_channels) == latent_num
        assert latent_num <= len(chs)
        assert len(scale_depth) == len(chs)
        assert len(kernel_size) == len(chs)
        assert len(dilation) == len(chs)
        decoder_head_in_channels = chs[0] + (0 if latent_num < len(chs) else latent_channels[-1])

        # self.encoder_head = ScaleBlock(in_ch, chs[0], activation, scale_depth[0], kernel_size[0], dilation[0], padding_mode='zeros')
        self.encoder_head = ConvBlock(in_ch, chs[0], activation, kernel_size[0], dilation[0], padding_mode='zeros')
        self.encoder = Encoder(chs, activation, scale_depth, kernel_size, dilation, padding_mode='zeros')
        self.decoder = Decoder(list(reversed(chs)), latent_num, activation, list(reversed(scale_depth)), list(reversed(kernel_size)), list(reversed(dilation)), padding_mode=padding_mode, latent_channels=latent_channels, latent_locks=latent_locks)
        # self.decoder_head = nn.Conv2d(decoder_head_in_channels, out_ch, kernel_size=1, dilation[0])
        self.decoder_head = ScaleBlock(decoder_head_in_channels, out_ch, activation, scale_depth[0], kernel_size[0], dilation[0], padding_mode=padding_mode)

        # self.posterior_encoder_head = ScaleBlock(in_ch+1, chs[0], activation, scale_depth[0], kernel_size[0], dilation[0], padding_mode='zeros')
        self.posterior_encoder_head = ConvBlock(in_ch+1, chs[0], activation, kernel_size[0], dilation[0], padding_mode='zeros')
        self.posterior_encoder = Encoder(chs, activation, scale_depth, kernel_size, dilation, padding_mode='zeros')

    def forward(self, x, y=None, times=1, first_channel_only=True, insert_from_postnet=False):
        f = self.encoder_head(x)
        f = self.encoder(f)
        
        outs, infodicts = [], []

        if y is None:  # Not Using Posterior Net
            for _ in range(times):
                o = self.decoder(f)
                o = self.decoder_head(o)
                outs.append(o)

        else:  # Using Posterior Net
            l = self.posterior_encoder_head(torch.cat([x, y], dim=1))
            l = self.posterior_encoder(l)

            for _ in range(times):
                o, infodict = self.decoder(f, l, insert_from_postnet)
                o = self.decoder_head(o)
                outs.append(o)
                infodicts.append(infodict)

        output = torch.stack(outs, dim=1)

        if first_channel_only is True:
            output = output[:,:,0]

        return output, infodicts


class SPUNet(nn.Module):
    def __init__(self, in_ch, chs, hlist, s, num_latent_channels=1, out_ch=1, activation="ReLU", scale_depth=1, kernel_size=3, padding_mode='circular'):
        super().__init__()
        assert hlist[-1] == 2 * num_latent_channels * s * s
        prob_chs = int(hlist[0] / s / s / num_latent_channels)

        # UNet
        self.encoder_head = ConvBlock(in_ch, chs[0], activation, kernel_size, padding_mode='zeros')
        self.encoder = Encoder(chs, activation, scale_depth, kernel_size, padding_mode='zeros')
        self.decoder = Decoder(list(reversed(chs)), 0, activation, scale_depth, kernel_size, padding_mode=padding_mode)

        # PriorNet
        self.prior_encoder_head = ConvBlock(in_ch, prob_chs, activation, kernel_size, padding_mode='zeros')
        self.prior_encoder = SimpleEncoder(hlist, num_latent_channels, s, activation)

        # PosteriorNet
        self.posterior_encoder_head = ConvBlock(in_ch+1, prob_chs, activation, kernel_size, padding_mode='zeros')
        self.posterior_encoder = SimpleEncoder(hlist, num_latent_channels, s, activation)

        # Output
        # self.latent_decoder = SimpleDecoder()
        self.decoder_head = ScaleBlock(chs[0]+num_latent_channels, out_ch, activation, scale_depth, kernel_size, padding_mode=padding_mode)

    
    def forward(self, x, y=None, times=1, return_sigma=False, insert_from_postnet=False):
        f_unet = self.encoder_head(x)
        f_unet = self.encoder(f_unet)
        f_unet = self.decoder(f_unet)

        f_prior = self.prior_encoder_head(x)
        prior_means, prior_logstds = self.prior_encoder(f_prior)
        
        if y is None:  # Not Using Posterior Net
            outs = []
            for _ in range(times):
                rands = torch.normal( mean=torch.zeros_like(prior_means), std=torch.ones_like(prior_logstds) )
                prior_latents = rands * torch.exp(prior_logstds) + prior_means

                o = torch.cat([f_unet, prior_latents], dim=1)
                o = self.decoder_head(o)
                outs.append(o)

            output = torch.stack(outs).squeeze(dim=2).transpose(0,1)

            if return_sigma is False and output.dim() == 5:
                output = output.permute(2,0,1,3,4)[0]

            return output

        else:  # Using Posterior Net
            f_post = self.posterior_encoder_head(torch.cat([x, y], dim=1))
            post_means, post_logstds = self.posterior_encoder(f_post)

            # Generate Latents
            rands = torch.normal( mean=torch.zeros_like(prior_means), std=torch.ones_like(prior_logstds) )
            prior_latents = rands * torch.exp(prior_logstds) + prior_means

            rands = torch.normal( mean=torch.zeros_like(post_means), std=torch.ones_like(post_logstds) )
            post_latents = rands * torch.exp(post_logstds) + post_means

            # Calculate KL
            prior_stds, post_stds = torch.exp(prior_logstds), torch.exp(post_logstds)

            kl = torch.log( prior_stds / post_stds )                             \
                + ( post_stds ** 2  +  (post_means - prior_means) ** 2 )         \
                    / 2 / prior_stds ** 2                                        \
                - 1 / 2
            kl = kl.sum(dim=(1,2,3)).mean()

            # Calculate Output
            if self.training is True or insert_from_postnet is True:  # Insert Latents from Posterior Net
                o = torch.cat([f_unet, post_latents], dim=1)
            else:  # Insert Latents from Prior Net
                o = torch.cat([f_unet, prior_latents], dim=1)
            
            o = self.decoder_head(o)

            # Items to return as well as the network's output
            infodict = {
                'kls': kl,
                'post_latents': [post_latents],
                'prior_latents': [prior_latents],
                'post_means': [post_means],
                'post_stds': [post_stds],
                'prior_means': [prior_means],
                'prior_stds': [prior_stds]
            }

            return o, infodict


# Optimization
class BetaConstant(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def step(self):
        return


class BetaLinearScheduler(nn.Module):
    def __init__(self, ascending_steps, constant_steps=0, max_beta=1.0, saturation_step=None):
        super().__init__()
        self.ascending_steps = ascending_steps
        self.constant_steps = constant_steps
        self.max_beta = max_beta
        if saturation_step is not None:
            self.saturation_gap = saturation_step
        else:
            self.saturation_gap = -1

        self.increment = max_beta / ascending_steps

        self.beta = 0.0
        self.state = 'ascend'
        self.s = 0

    def step(self):
        if self.state == 'ascend':
            self.beta += self.increment
            self.s += 1
            if self.s == self.ascending_steps:
                self.state = 'constant'
                self.s = 0
        
        elif self.state == 'constant':
            self.s += 1
            if self.s == self.constant_steps:
                self.state = 'ascend'
                self.s = 0
                self.beta = 0.0
        
        self.saturation_gap -= 1
        if self.saturation_gap == 0:
            self.state = 'saturated'
            self.beta = self.max_beta

        return


# Loss Functions & Utils
def sample_gumbel(shape, eps=1e-20, device='cpu'):
    return - torch.log(
        - torch.log(
            torch.rand(shape, device=device) + eps
        )
        + eps
     )


def topk_mask(score, k, deterministic=False):
    with torch.no_grad():
        score_flat = score.clone().reshape(-1)
        k_int = int( score_flat.shape[0] * k )
        assert 1 <= k_int <= score_flat.shape[0]

        # Normalize score and Add Gumbel noise (if not deterministic)
        score_flat /= score_flat.sum()
        log_score_flat = torch.log(score_flat)
        if not deterministic:  # Gumbel trick
            log_score_flat += sample_gumbel(log_score_flat.shape, device=score.device)
        
        # Sample indices of top k elements
        _, indices = torch.topk(log_score_flat, k_int)

        # Create top k mask
        topk_mask = torch.zeros_like(score_flat, dtype=score.dtype, device=score.device)        
        topk_mask.scatter_add_(dim=0, index=indices, src=torch.ones(k_int, dtype=score.dtype, device=score.device))
        topk_mask = topk_mask.to(int)

        return topk_mask.reshape_as(score)


class TopkMaskedMSELoss(nn.Module):
    def __init__(self, k, deterministic):
        super().__init__()
        self.k = k
        self.deterministic = deterministic
        self.mse = MSELoss(reduction='none')
        
        self.last_loss = None

 
    def forward(self, yhat, y):        
        mse = self.mse(yhat, y)
        mask = topk_mask(mse, self.k, self.deterministic)
        loss = mask * mse / self.k

        self.last_loss = {
            'mse': mse,
            'mask': mask,
            'loss': loss
        }

        return loss


class ELBOLoss(nn.Module):
    def __init__(self, beta=None, k=None, topk_deterministic=False):
        super().__init__()
        if beta is None:
            beta = BetaConstant(1.0)
        self.beta_scheduler = beta
        
        if k is None:
            self.mse = MSELoss(reduction='none')
        else:
            assert 0.0 < k <= 1.0
            self.mse = TopkMaskedMSELoss(k=k, deterministic=topk_deterministic)

        self.last_loss = None

 
    def forward(self, yhat, y, kls, logstd2=None):        
        if logstd2 is None:
            rec_loss = self.mse(yhat, y).sum(dim=(1,2)).mean()
        
        else:
            rec_term1 = ( self.mse(yhat, y) / logstd2.exp() ).sum(dim=(1,2)).mean()
            rec_term2 = logstd2.sum(dim=(1,2)).mean()
            rec_loss = rec_term1 + rec_term2
        
        loss = rec_loss + self.beta_scheduler.beta * torch.sum(kls)

        self.last_loss = {
            'rec_term1': rec_term1 if logstd2 is not None else None,
            'rec_term2': rec_term2 if logstd2 is not None else None,
            'reconstruction_term': rec_loss,
            'kl_term': self.beta_scheduler.beta * torch.sum(kls),
            'loss': loss
        }

        return loss


class GECOLoss(nn.Module):
    def __init__(self, kappa, decay=0.9, update_rate=0.01, device='cpu', k=None, topk_deterministic=False):
        super(GECOLoss, self).__init__()
        self.kappa = kappa
        self.decay = decay
        self.update_rate = update_rate
        self.device = device

        self.log_lamda = torch.nn.Parameter(torch.FloatTensor([0.0]), requires_grad=False)
        self.rec_constraint_ma = None

        if k is None:
            self.mse = MSELoss(reduction='none')
        else:
            assert 0.0 < k <= 1.0
            self.mse = TopkMaskedMSELoss(k=k, deterministic=topk_deterministic)

        self.last_loss = None


    def update_rec_constraint_ma(self, cons):
        if self.rec_constraint_ma is None:
            self.rec_constraint_ma = torch.FloatTensor([cons]).to(self.device)
        else:
            self.rec_constraint_ma = self.decay * self.rec_constraint_ma.detach() + (1-self.decay) * cons
    

    def forward(self, yhat, y, kls, lr=None, logstd2=None):
        if logstd2 is None:
            rec_loss = self.mse(yhat, y).sum(dim=(1,2)).mean()
        
        else:
            rec_term1 = ( self.mse(yhat, y) / logstd2.exp() ).sum(dim=(1,2)).mean()
            rec_term2 = logstd2.sum(dim=(1,2)).mean()
            rec_loss = rec_term1 + rec_term2
        
        rec_constraint = rec_loss - self.kappa

        # Update EMA
        if self.training is True:
            self.update_rec_constraint_ma(rec_constraint)

        # Calculate Loss
        rec_constraint_ma = rec_constraint + (self.rec_constraint_ma - rec_constraint).detach()
        # lamda = nn.functional.softplus(self.log_lamda)
        lamda = torch.exp(self.log_lamda)
        loss = lamda * rec_constraint_ma + torch.sum(kls)

        self.last_loss = {
            'rec_term1': rec_term1 if logstd2 is not None else None,
            'rec_term2': rec_term2 if logstd2 is not None else None,
            'reconstruction_term': lamda * rec_constraint_ma,
            'kl_term': torch.sum(kls),
            'loss': loss
        }

        # Step Lambda
        if self.training is True:
            with torch.no_grad():
                self.log_lamda += self.update_rate * lr * rec_constraint_ma

        return loss


class AdvLoss(nn.Module):
    def __init__(self, pixels=1.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.pixels = pixels
        self.beta = beta
        self.gamma = gamma

        self.mse = MSELoss(reduction='none')
        self.bce = BCEWithLogitsLoss()
 
    def forward(self, preds, truths, yhat, y, kls=0, update=True):
        reconstruction = self.mse(preds, truths).sum(dim=(1,2,3)).mean()
        bce = self.bce(yhat, y)

        loss = (reconstruction + self.beta * torch.sum(kls)) / self.pixels + self.gamma * bce

        return loss, reconstruction, bce


# Weight Initializers
def ortho_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.orthogonal_(m.weight.data)
        nn.init.normal_(m.bias.data, mean=0, std=0.001)
    elif classname.find('ConvTranspose2d') != -1:
        nn.init.orthogonal_(m.weight.data)
        nn.init.normal_(m.bias.data, mean=0, std=0.001)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.uniform_(m.weight.data, a=-1, b=1)
        nn.init.normal_(m.bias.data, mean=0, std=0.001)



# Discriminators
class PureConvClassifier32(nn.Module):
    def __init__(self):
        super(PureConvClassifier32, self).__init__()
        self.ndf = 64
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=self.ndf * 2, out_channels=self.ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=self.ndf * 4, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, input):
        return self.main(input)


class PureConvClassifier24(nn.Module):
    def __init__(self):
        super(PureConvClassifier24, self).__init__()
        self.ndf = 64
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=self.ndf, kernel_size=4, stride=2, padding=5, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=self.ndf * 2, out_channels=self.ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=self.ndf * 4, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, input):
        return self.main(input)


class PureConvClassifier16(nn.Module):
    def __init__(self):
        super(PureConvClassifier16, self).__init__()
        self.ndf = 64
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=self.ndf * 2, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, input):
        return self.main(input)


class PureConvClassifier8(nn.Module):
    def __init__(self):
        super(PureConvClassifier8, self).__init__()
        self.ndf = 64
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=self.ndf, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, input):
        return self.main(input)