import time
from random import randrange

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import tracemalloc
import pickle

def train_model(args, model, dataloader, criterion, optimizer, lr_scheduler, writer, device='cpu', val_dataloader=None, start_time=None): 
    history = {
        'training_time(min)': None,
        
        'minibatch_loss_total': [],
        'minibatch_loss_reconstruction': [],
        'minibatch_loss_kl_term': [],
        'minibatch_loss_kl': [[] for _ in range(args.latent_num)],
    }
    
    if args.loss_type.lower() == 'geco':
        history.update( {
            'lambda': []
        } )

    if val_dataloader is not None:
        history.update( {
            'mean_val_loss_total': [],
            'mean_val_loss_reconstruction': [],
            'mean_val_loss_kl_term': [],
            'mean_val_loss_kl': [[] for _ in range(args.latent_num)]
        } )

        val_minibatches = len(val_dataloader)


    def record_history(idx, loss_dict, type='train'):
        prefix, prename = 'minibatch_', 'Minibatch '
        if type == 'val':
            prefix = 'mean_val_'
            prename = 'Mean Validation '

        # Logging For TensorBoard
        loss_per_pixel = loss_dict['loss'].item() / args.pixels
        reconstruction_per_pixel = loss_dict['reconstruction_term'].item() / args.pixels
        kl_term_per_pixel = loss_dict['kl_term'].item() / args.pixels
        kl_per_pixel = [ loss_dict['kls'][v].item() / args.pixels for v in range(args.latent_num) ]

        ## Total Loss
        _dict = {   'total': loss_per_pixel,
                    'kl term': kl_term_per_pixel, 
                    'reconstruction': reconstruction_per_pixel  }
        writer.add_scalars(prename + 'Loss Curve', _dict, idx)
        
        ## Reconstruction Term Decomposition
        if args.std_type is not None:
            _dict = {   'reconstruction term 1': loss_dict['rec_term1'].item() / args.pixels,
                        'reconstruction term 2': loss_dict['rec_term2'].item() / args.pixels   }
            writer.add_scalars(prename + 'Loss Curve (Reconstruction)', _dict, idx)

        ## KL Term Decomposition
        _dict = { 'sum': sum(kl_per_pixel) }
        _dict.update( { 'scale {}'.format(v): kl_per_pixel[v] for v in range(args.latent_num) } )
        writer.add_scalars(prename + 'Loss Curve (K-L)', _dict, idx)

        if type == 'train' and args.loss_type.lower() == 'geco':
            writer.add_scalar('Lagrangian Coefficient', criterion.log_lamda.exp().item(), idx)


        # # Logging For Output
        # history[prefix + 'loss_total'].append(loss_per_pixel)
        # history[prefix + 'loss_kl_term'].append(kl_term_per_pixel)
        # history[prefix + 'loss_reconstruction'].append(reconstruction_per_pixel)
        # for v in range(args.latent_num):
        #     history[prefix + 'loss_kl'][v].append(kl_per_pixel[v])
        # if type == 'train' and args.loss_type.lower() == 'geco':
        #     history['lambda'].append( criterion.log_lamda.exp().item() )

    val_images, val_truths = next(iter(val_dataloader))
    val_images, val_truths = val_images[:16], val_truths[:16]
    truth_grid = make_grid(val_truths, nrow=4, pad_value=val_truths.min().item())
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(truth_grid[0])
    ax.set_axis_off()
    fig.tight_layout()
    writer.add_figure('Validation Images / Ground Truth', fig)
    val_images_selection = val_images.to(device)
    val_truths_selection = val_truths.to(device)
    
    last_time_checkpoint = start_time
    for e in range(args.epochs):
        for mb, (images, truths) in enumerate(dataloader):
            idx = e*len(dataloader) + mb+1

            # Initialization
            criterion.train()
            model.train()
            model.zero_grad()
            images, truths = images.to(device), truths.to(device)

            # Train One Step
            
            ## Get Predictions and Prepare for Loss Calculation
            if args.std_type is None:
                preds, infodicts = model(images, truths)
                preds, infodict = preds[:,0], infodicts[0]

            elif args.std_type.lower() == 'predict':
                preds, infodicts = model(images, truths, first_channel_only=False)
                preds, logstd2, infodict = preds[:,0,0], preds[:,0,1], infodicts[0]

            elif args.std_type.lower() == 'sample':
                preds, infodicts = model(images, truths, times=args.std_sample_num)
                with torch.no_grad():
                    logstd2 = torch.log( torch.var(preds.detach(), dim=1, unbiased=True) )
                k = randrange(args.std_sample_num)
                preds, infodict = preds[:,k], infodicts[k]

            truths = truths.squeeze(dim=1)


            ## Calculate Loss
            if args.loss_type.lower() == 'elbo':
                if args.std_type is None:
                    loss = criterion(preds, truths, kls=infodict['kls'])
                
                elif args.std_type.lower() in ['predict', 'sample']:
                    loss = criterion(preds, truths, kls=infodict['kls'], logstd2=logstd2)

            elif args.loss_type.lower() == 'geco':
                if args.std_type is None:
                    loss = criterion(preds, truths, kls=infodict['kls'], lr=lr_scheduler.get_last_lr()[0])

                elif args.std_type.lower() in ['predict', 'sample']:
                    loss = criterion(preds, truths, kls=infodict['kls'], lr=lr_scheduler.get_last_lr()[0], logstd2=logstd2)


            ## Backpropagate
            loss.backward()             # Calculate Gradients
            optimizer.step()            # Update Weights
            

            ## Step Beta Scheduler
            if args.loss_type.lower() == 'elbo':
                criterion.beta_scheduler.step()


            # Record Train History
            loss_dict = criterion.last_loss.copy()
            loss_dict.update( { 'kls': infodict['kls'] } )
            record_history(idx, loss_dict)
            
            # Record Beta
            if args.loss_type.lower() == 'elbo':
                writer.add_scalar('Beta', criterion.beta_scheduler.beta, idx)
            
            # Validation
            if idx % args.val_period == 0:
                criterion.eval()
                model.eval()

                # Show Sample Validation Images
                with torch.no_grad():
                    val_preds = model(val_images_selection)[0]
                    out_grid = make_grid(val_preds, nrow=4, pad_value=val_preds.min().item())

                    fig, ax = plt.subplots(figsize=(6,6))
                    ax.imshow(out_grid[0].cpu())
                    ax.set_axis_off()
                    fig.tight_layout()
                    writer.add_figure('Validation Images / Prediction', fig, idx)


                # Calculate Validation Loss
                mean_val_loss, mean_val_reconstruction_term, mean_val_kl_term = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
                mean_val_rec_term1, mean_val_rec_term2 = torch.zeros(1, device=device), torch.zeros(1, device=device)
                mean_val_kl = torch.zeros(args.latent_num, device=device)
                with torch.no_grad():
                    for _, (val_images, val_truths) in enumerate(val_dataloader):
                        val_images, val_truths = val_images.to(device), val_truths.to(device)


                        ## Get Predictions and Prepare for Loss Calculation
                        if args.std_type is None:
                            preds, infodicts = model(val_images, val_truths)
                            preds, infodict = preds[:,0], infodicts[0]

                        elif args.std_type.lower() == 'predict':
                            preds, infodicts = model(val_images, val_truths, first_channel_only=False)
                            preds, logstd2, infodict = preds[:,0,0], preds[:,0,1], infodicts[0]

                        elif args.std_type.lower() == 'sample':
                            preds, infodicts = model(val_images, val_truths, times=args.std_sample_num)
                            logstd2 = torch.log( torch.var(preds.detach(), dim=1, unbiased=True) )
                            k = randrange(args.std_sample_num)
                            preds, infodict = preds[:,k], infodicts[k]

                        val_truths = val_truths.squeeze(dim=1)


                        ## Calculate Loss
                        if args.loss_type.lower() == 'elbo':
                            if args.std_type is None:
                                loss = criterion(preds, val_truths, kls=infodict['kls'])
                            
                            elif args.std_type.lower() in ['predict', 'sample']:
                                loss = criterion(preds, val_truths, kls=infodict['kls'], logstd2=logstd2)

                        elif args.loss_type.lower() == 'geco':
                            if args.std_type is None:
                                loss = criterion(preds, val_truths, kls=infodict['kls'], lr=lr_scheduler.get_last_lr()[0])

                            elif args.std_type.lower() in ['predict', 'sample']:
                                loss = criterion(preds, val_truths, kls=infodict['kls'], lr=lr_scheduler.get_last_lr()[0], logstd2=logstd2)


                        mean_val_loss += loss
                        mean_val_reconstruction_term += criterion.last_loss['reconstruction_term']
                        mean_val_kl_term += criterion.last_loss['kl_term']
                        mean_val_kl += infodict['kls']
                        if args.std_type is not None:
                            mean_val_rec_term1 += criterion.last_loss['rec_term1']
                            mean_val_rec_term2 += criterion.last_loss['rec_term2']
                    

                    mean_val_loss /= val_minibatches
                    mean_val_reconstruction_term /= val_minibatches
                    mean_val_kl_term /= val_minibatches
                    mean_val_kl /= val_minibatches
                    mean_val_rec_term1 /= val_minibatches
                    mean_val_rec_term2 /= val_minibatches


                # Record Validation History
                loss_dict = {
                    'loss': mean_val_loss,
                    'reconstruction_term': mean_val_reconstruction_term,
                    'kl_term': mean_val_kl_term,
                    'kls': mean_val_kl,
                    'rec_term1': mean_val_rec_term1,
                    'rec_term2': mean_val_rec_term2
                }
                record_history(idx, loss_dict, type='val')
        
        
        # Report Epoch Completion
        time_checkpoint = time.time()
        epoch_time = (time_checkpoint - last_time_checkpoint) / 60
        total_time = (time_checkpoint - start_time) / 60
        print('Epoch {}/{} done in {:.1f} minutes. \t\t\t\t Total time: {:.1f} minutes'.format(e+1, args.epochs, epoch_time, total_time))
        last_time_checkpoint = time_checkpoint

        # Take and Save Memory Snapshot
        if (e+1) % 20 == 0:
            snapshot = tracemalloc.take_snapshot()
            pickle.dump( snapshot, open('runs/{}/memory/{}.p'.format(args.stamp, e+1), 'wb') )
        
        # Save Model and Loss
        if (e+1) % args.save_period == 0 and (e+1) != args.epochs:
            torch.save(model, 'runs/{}/model{}.pth'.format(args.stamp, e+1))
            torch.save(criterion, 'runs/{}/loss{}.pth'.format(args.stamp, e+1))
        
        # Step Learning Rate
        writer.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], e)
        lr_scheduler.step()


    return history