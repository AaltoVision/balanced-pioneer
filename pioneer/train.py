import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torchvision import utils

from pioneer.model import Generator, Discriminator, SpectralNormConv2d

from datetime import datetime
import random
import copy

import os

from pioneer import config
from pioneer import utils
from pioneer import data
from pioneer import evaluate

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from torch.nn import functional as F

from pioneer.training_scheduler import TrainingScheduler

args   = config.get_config()
writer = None

def batch_size(reso):
    if args.gpu_count == 1:
        save_memory = False
        if not save_memory:
            batch_table = {4:128, 8:128, 16:128, 32:64, 64:32, 128:32, 256:32, 512:4, 1024:1}
        else:
            batch_table = {4:128, 8:128, 16:128, 32:32, 64:16, 128:4, 256:2, 512:2, 1024:1}
    elif args.gpu_count == 2:
        batch_table = {4:256, 8:256, 16:256, 32:128, 64:64, 128:32, 256:32, 512:8, 1024:2}
    elif args.gpu_count == 4:
        batch_table = {4:256, 8:256, 16:256, 32:128, 64:64, 128:32, 256:32, 512:16, 1024:4}
    elif args.gpu_count == 8:
        batch_table = {4:512, 8:512, 16:512, 32:256, 64:256, 128:128, 256:64, 512:32, 1024:8}
    else:
        assert(False)
    
    return batch_table[reso]

def batch_size_by_phase(phase):
    return batch_size(4 * 2 ** phase)

def loadSNU(path):
    if os.path.exists(path):
        us_list = torch.load(path)

        print("Loading {} spectral norm layers from SNU...".format(len(SpectralNormConv2d.spectral_norm_layers)))
        for layer_i, layer in enumerate(SpectralNormConv2d.spectral_norm_layers):
            setattr(layer, 'weight_u', us_list[layer_i])
    else:
        print("Warning! No SNU found.")

def saveSNU(path):
    us = []
    for layer in SpectralNormConv2d.spectral_norm_layers:
        us += [getattr(layer, 'weight_u')]
    torch.save(us, path)

class Session:
    def __init__(self):
        # Note: 4 requirements for sampling from pre-existing models:
        # 1) Ensure you save and load both multi-gpu versions (DataParallel) or both not.
        # 2) Ensure you set the same phase value as the pre-existing model and that your local and global alpha=1.0 are set
        # 3) Sample from the g_running, not from the latest generator
        # 4) You may need to warm up the g_running by running evaluate.reconstruction_dryrun() first

        self.alpha = -1
        self.sample_i = min(args.start_iteration, 0)
        self.phase = args.start_phase

        self.generator = nn.DataParallel( Generator(args.nz+1, args.n_label).to(device=args.device) )
        self.g_running = nn.DataParallel( Generator(args.nz+1, args.n_label).to(device=args.device) )
        self.encoder   = nn.DataParallel( Discriminator(nz = args.nz+1, n_label = args.n_label, binary_predictor = args.train_mode == config.MODE_GAN).to(device=args.device) )

        print("Using ", torch.cuda.device_count(), " GPUs!")

        self.reset_opt()

        print('Session created.')

    def reset_opt(self):
        lrG = args.lr
        lrE = args.lr
        beta2 = 0.99

        self.optimizerG = optim.Adam(self.generator.parameters(), lrG, betas=(0.0, 0.9))
        self.optimizerD = optim.Adam(self.encoder.parameters(), lrE, betas=(0.0, 0.9)) # includes all the encoder parameters...

    def save_all(self, path):
        torch.save({'G_state_dict': self.generator.state_dict(),
                    'D_state_dict': self.encoder.state_dict(),
                    'G_running_state_dict': self.g_running.state_dict(),
                    'optimizerD': self.optimizerD.state_dict(),
                    'optimizerG': self.optimizerG.state_dict(),
                    'iteration': self.sample_i,
                    'phase': self.phase,
                    'alpha': self.alpha},
                path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.sample_i = int(checkpoint['iteration'])
        

        self.generator.load_state_dict(checkpoint['G_state_dict'])
        self.g_running.load_state_dict(checkpoint['G_running_state_dict'])
        self.encoder.load_state_dict(checkpoint['D_state_dict'])

        if args.reset_optimizers <= 0:
            self.optimizerD.load_state_dict(checkpoint['optimizerD'])
            self.optimizerG.load_state_dict(checkpoint['optimizerG'])
            print("Reloaded old optimizers")
        else:
            print("Despite loading the state, we reset the optimizers.")

        self.alpha = checkpoint['alpha']
        self.phase = int(checkpoint['phase'])
        if args.start_phase > 0: #If the start phase has been manually set, try to actually use it (e.g. when have trained 64x64 for extra rounds and then turning the model over to 128x128)
            self.phase = min(args.start_phase, self.phase)
            print("Use start phase: {}".format(self.phase))
        if self.phase > args.max_phase:
            print('Warning! Loaded model claimed phase {} but max_phase={}'.format(self.phase, args.max_phase))
            self.phase = args.max_phase

    def create(self):
        if args.start_iteration <= 0:
            args.start_iteration = 1
            if args.no_progression:
                self.sample_i = args.start_iteration = int( (args.max_phase + 0.5) * args.images_per_stage ) # Start after the fade-in stage of the last iteration
                args.force_alpha = 1.0
                print("Progressive growth disabled. Setting start step = {} and alpha = {}".format(args.start_iteration, args.force_alpha))
        else:
            reload_from = '{}/checkpoint/{}_state'.format(args.save_dir, str(args.start_iteration).zfill(6)) #e.g. '604000' #'600000' #latest'   
            reload_from_SNU = '{}/checkpoint/{}_SNU'.format(args.save_dir, str(args.start_iteration).zfill(6))
            print(reload_from)
            if os.path.exists(reload_from):
                self.load(reload_from)
                print("Loaded {}".format(reload_from))
                print("Iteration asked {} and got {}".format(args.start_iteration, self.sample_i))               
                if args.load_SNU:
                    loadSNU(reload_from_SNU)
                else:
                    print("Ignore SNU")
            else:
                assert(not args.testonly)
                self.sample_i = args.start_iteration
                print('Start from iteration {}'.format(self.sample_i))

        self.g_running.train(False)

        if args.force_alpha >= 0.0:
            self.alpha = args.force_alpha

        if not args.testonly:
            accumulate(self.g_running, self.generator, 0)

def setup():
    config.init()
    
    utils.make_dirs()
    if not args.testonly:
        config.log_args(args)

    if args.use_TB:
        from dateutil import tz
        from tensorboardX import SummaryWriter

        dt = datetime.now(tz.gettz('Europe/Helsinki')).strftime(r"%y%m%d_%H%M")
        global writer
        writer = SummaryWriter("{}/{}/{}".format(args.summary_dir, args.save_dir, dt))

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)   

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

def get_grad_penalty(discriminator, real_image, fake_image, step, alpha):
    """ Used in WGAN-GP version only. """
    eps = torch.rand(batch_size_by_phase(step), 1, 1, 1).to(device=args.device) #.cuda()

    if eps.size(0) != real_image.size(0) or eps.size(0) != fake_image.size(0):
        # If end-of-batch situation, we restrict other vectors to matcht the number of training images available.
        eps = eps[:real_image.size(0)]
        fake_image = fake_image[:real_image.size(0)]

    x_hat = eps * real_image.data + (1 - eps) * fake_image.data
    x_hat = Variable(x_hat, requires_grad=True)

    if args.train_mode == config.MODE_GAN: # Regular GAN mode
        hat_predict, _ = discriminator(x_hat, step, alpha, args.use_ALQ)
        grad_x_hat = grad(
            outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0] #.item()?
    else:
        hat_z = discriminator(x_hat, step, alpha, args.use_ALQ)
        # KL_fake: \Delta( e(g(Z)) , Z ) -> max_e
        KL_maximizer = KLN01Loss(direction=args.KL, minimize=False)
        KL_fake = KL_maximizer(hat_z) * args.fake_D_KL_scale
        grad_x_hat = grad(
            outputs=KL_fake.sum(), inputs=x_hat, create_graph=True)[0] #.item()?

    # Push the gradients of the interpolated samples towards 1
    grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1)
                        .norm(2, dim=1) - 1)**2).mean()
    grad_penalty = 10 * grad_penalty
    return grad_penalty

def D_prediction_of_G_output(generator, encoder, step, alpha):
    # To use labels, enable here and elsewhere:
    #label = Variable(torch.ones(batch_size_by_phase(step), args.n_label)).cuda()
    #               label = Variable(
    #                    torch.multinomial(
    #                        torch.ones(args.n_label), args.batch_size, replacement=True)).cuda()

    myz = Variable(torch.randn(batch_size_by_phase(step), args.nz)).to(device=args.device) #.cuda(async=(args.gpu_count>1))
    myz = utils.normalize(myz)

    fake_image = generator(myz, None, step, alpha)
    fake_predict, _ = encoder(fake_image, step, alpha, args.use_ALQ)

    loss = fake_predict.mean()
    return loss, fake_image

class KLN01Loss(torch.nn.Module): #Adapted from https://github.com/DmitryUlyanov/AGE

    def __init__(self, direction, minimize):
        super(KLN01Loss, self).__init__()
        self.minimize = minimize
        assert direction in ['pq', 'qp'], 'direction?'

        self.direction = direction

    def forward(self, samples):

        assert samples.nelement() == samples.size(1) * samples.size(0), '?'

        samples = samples.view(samples.size(0), -1)

        self.samples_var = utils.var(samples)
        self.samples_mean = samples.mean(0)

        samples_mean = self.samples_mean
        samples_var = self.samples_var

        if self.direction == 'pq':
            t1 = (1 + samples_mean.pow(2)) / (2 * samples_var.pow(2))
            t2 = samples_var.log()

            KL = (t1 + t2 - 0.5).mean()
        else:
            # In the AGE implementation, there is samples_var^2 instead of samples_var^1
            t1 = (samples_var + samples_mean.pow(2)) / 2
            # In the AGE implementation, this did not have the 0.5 scaling factor:
            t2 = -0.5*samples_var.log()

            KL = (t1 + t2 - 0.5).mean()

        if not self.minimize:
            KL *= -1

        return KL

def encoder_train(session, real_image, generatedImagePool, batch_N, match_x, stats, kls, margin):
    encoder = session.encoder
    generator = session.generator

    encoder.zero_grad()

    x = Variable(real_image).to(device=args.device) #.cuda(async=(args.gpu_count>1))
    KL_maximizer = KLN01Loss(direction=args.KL, minimize=False)
    KL_minimizer = KLN01Loss(direction=args.KL, minimize=True)

    e_losses = []

    real_z = encoder(x, session.phase, session.alpha, args.use_ALQ)

    if args.use_real_x_KL:
        # KL_real: - \Delta( e(X) , Z ) -> max_e
        KL_real = KL_minimizer(real_z) * args.real_x_KL_scale

        stats['real_mean'] = KL_minimizer.samples_mean.data.mean()
        stats['real_var'] = KL_minimizer.samples_var.data.mean()
        stats['KL_real'] = KL_real.data.item()
        kls = "{0:.3f}".format(stats['KL_real'])

    recon_x = generator(real_z, None, session.phase, session.alpha)
    if args.use_loss_x_reco:
        # match_x: E_x||g(e(x)) - x|| -> min_e
        err = utils.mismatch(recon_x, x, args.match_x_metric) * match_x
        stats['x_reconstruction_error'] = err.data.item()
        if args.boostreco > 0:
             err = err * (1 + max(0, (session.phase+session.alpha-5)*args.boostreco)) #Start boosting at 128x128
        e_losses.append(err)

    args.use_wpgan_grad_penalty = False
    grad_penalty = 0.0
    
    if args.use_loss_fake_D_KL:
        # TODO: The following codeblock is essentially the same as the KL_minimizer part on G side. Unify

        z = Variable( torch.FloatTensor(batch_N, args.nz, 1, 1) ).to(device=args.device) #.cuda(async=(args.gpu_count>1))
        utils.populate_z(z, args.nz+args.n_label, args.noise, batch_N)
        
        with torch.no_grad():
            fake = generator(z, None, session.phase, session.alpha)

        if session.alpha >= 1.0:
            fake = generatedImagePool.query(fake.data)

        fake.requires_grad_()

        # e(g(Z))
        egz = encoder(fake, session.phase, session.alpha, args.use_ALQ)

        # KL_fake: \Delta( e(g(Z)) , Z ) -> max_e

        if args.noSCHED:
            margin = args.m

        m = margin

        KL_fake = KL_maximizer(egz) * args.fake_D_KL_scale

        if m > 0.0:
            KL_loss = torch.max(-torch.ones_like(KL_fake) * m, KL_real + KL_fake) # KL_fake is always negative with abs value typically larger than KL_real. Hence, the sum is negative, and must be gapped so that the minimum is the negative of the margin.
            if args.realm:
                KL_loss = KL_loss + KL_real
        else:
            KL_loss = KL_real + KL_fake

        e_losses.append(KL_loss)

        stats['fake_mean'] = KL_maximizer.samples_mean.data.mean()
        stats['fake_var'] = KL_maximizer.samples_var.data.mean()
        stats['KL_fake'] = -KL_fake.data.item()
        stats['KL_loss'] = KL_loss.item()
        kls = "{0}/{1:.3f}/{2:.4f}".format(kls, stats['KL_fake'], stats['KL_loss'])

        if args.use_wpgan_grad_penalty:
            grad_penalty = get_grad_penalty(encoder, x, fake, session.phase, session.alpha)

    # Update e
    if len(e_losses) > 0:
        e_loss = sum(e_losses)                
        stats['E_loss'] = np.float32(e_loss.data.item())
        e_loss.backward()

        if args.use_wpgan_grad_penalty:
            grad_penalty.backward()
            stats['Grad_penalty'] = grad_penalty.data

    session.optimizerD.step()

    return kls

def KL_of_encoded_G_output(generator, encoder, z, batch_N, session):
    KL_minimizer = KLN01Loss(direction=args.KL, minimize=True)
    utils.populate_z(z, args.nz+args.n_label, args.noise, batch_N)
    fake = generator(z, None, session.phase, session.alpha)
    
    egz = encoder(fake, session.phase, session.alpha, args.use_ALQ)
    # KL_fake: \Delta( e(g(Z)) , Z ) -> min_g
    return egz, KL_minimizer(egz) * args.fake_G_KL_scale, z


def decoder_train(session, batch_N, stats, kls):
    session.generator.zero_grad()
    
    g_losses = []
    z = Variable( torch.FloatTensor(batch_N, args.nz, 1, 1) ).to(device=args.device) #.cuda(async=(args.gpu_count>1))

    egz, kl, z = KL_of_encoded_G_output(session.generator, session.encoder, z, batch_N, session)

    if args.use_loss_KL_z:
        g_losses.append(kl) # G minimizes this KL
        stats['KL(Phi(G))'] = kl.data.item()
        kls = "{0}/{1:.3f}".format(kls, stats['KL(Phi(G))'])

    if args.use_loss_z_reco:
        z_diff = utils.mismatch(egz, z, args.match_z_metric) * args.match_z # G tries to make the original z and encoded z match
        g_losses.append(z_diff)                

    if len(g_losses) > 0:
        loss = sum(g_losses)
        stats['G_loss'] = np.float32(loss.data.item())
        loss.backward()
    
    session.optimizerG.step()

    if args.train_mode == config.MODE_CYCLIC:
        if args.use_loss_z_reco:
            stats['z_reconstruction_error'] = z_diff.data.item()

    return kls


def train(generator, encoder, g_running, train_data_loader, test_data_loader, session, total_steps, train_mode, sched):
    pbar = tqdm(initial=session.sample_i, total = total_steps)

    benchmarking = False
  
    match_x = args.match_x
    generatedImagePool = None

    refresh_dataset   = True
    refresh_imagePool = True

    # After the Loading stage, we cycle through successive Fade-in and Stabilization stages

    batch_count = 0

    reset_optimizers_on_phase_start = False

    # TODO Unhack this (only affects the episode count statistics anyway):
    if args.data != 'celebaHQ':
        epoch_len = len(train_data_loader(1,4).dataset)
    else:
        epoch_len = train_data_loader._len['data4x4']

    if args.step_offset != 0:
        if args.step_offset == -1:
            args.step_offset = session.sample_i
        print("Step offset is {}".format(args.step_offset))
        session.phase += args.phase_offset
        session.alpha = 0.0

    while session.sample_i < total_steps:
        #######################  Phase Maintenance ####################### 

        sched.update(session.sample_i)
        sample_i_current_stage = sched.get_iteration_of_current_phase(session.sample_i)

        if sched.phaseChangedOnLastUpdate:
            match_x = args.match_x # Reset to non-matching phase
            refresh_dataset = True
            refresh_imagePool = True # Reset the pool to avoid images of 2 different resolutions in the pool
            if reset_optimizers_on_phase_start:
                utils.requires_grad(generator)
                utils.requires_grad(encoder)
                generator.zero_grad()
                encoder.zero_grad()
                session.reset_opt()
                print("Optimizers have been reset.")                

        reso = 4 * 2 ** session.phase

        # If we can switch from fade-training to stable-training
        if sample_i_current_stage >= args.images_per_stage/2:
            if session.alpha < 1.0:
                refresh_dataset = True # refresh dataset generator since no longer have to fade
            match_x = args.match_x * args.matching_phase_x
        else:
            match_x = args.match_x

        session.alpha = min(1, sample_i_current_stage * 2.0 / args.images_per_stage) # For 100k, it was 0.00002 = 2.0 / args.images_per_stage

        if refresh_dataset:
            train_dataset = data.Utils.sample_data2(train_data_loader, batch_size(reso), reso, session)
            refresh_dataset = False
            print("Refreshed dataset. Alpha={} and iteration={}".format(session.alpha, sample_i_current_stage))
        if refresh_imagePool:
            imagePoolSize = 200 if reso < 256 else 100
            generatedImagePool = utils.ImagePool(imagePoolSize) #Reset the pool to avoid images of 2 different resolutions in the pool
            refresh_imagePool = False
            print('Image pool created with size {} because reso is {}'.format(imagePoolSize, reso))

        ####################### Training init ####################### 
             
        stats = {}

        try:
            real_image, _ = next(train_dataset)              
        except (OSError, StopIteration):
            train_dataset = data.Utils.sample_data2(train_data_loader, batch_size(reso), reso, session)
            real_image, _ = next(train_dataset)

        ####################### DISCRIMINATOR / ENCODER ###########################

        utils.switch_grad_updates_to_first_of(encoder, generator)
        kls = ""

        if train_mode == config.MODE_GAN:
            one = torch.FloatTensor([1]).to(device=args.device) #.cuda(async=(args.gpu_count>1))
            x = Variable(real_image).to(device=args.device) #.cuda(async=(args.gpu_count>1))
            encoder.zero_grad()

            # Discriminator for real samples
            real_predict, _ = encoder(x, session.phase, session.alpha, args.use_ALQ)
            real_predict = real_predict.mean() \
                - 0.001 * (real_predict ** 2).mean()
            real_predict.backward(-one) # Towards 1

            # (1) Generator => D. Identical to (2) see below
            
            fake_predict, fake_image = D_prediction_of_G_output(generator, encoder, session.phase, session.alpha)
            fake_predict.backward(one)

            # Grad penalty

            grad_penalty = get_grad_penalty(encoder, x, fake_image, session.phase, session.alpha)
            grad_penalty.backward()

            del x, one

        elif train_mode == config.MODE_CYCLIC:
            kls = encoder_train(session, real_image, generatedImagePool, batch_size(reso), match_x, stats, kls, margin = sched.m)

        ######################## GENERATOR / DECODER #############################

        if (batch_count + 1) % args.n_critic == 0:
            utils.switch_grad_updates_to_first_of(generator, encoder)
            
            for _ in range(args.n_generator):               
                if train_mode == config.MODE_GAN:
                    generator.zero_grad()
                    fake_predict, _ = D_prediction_of_G_output(generator, encoder, session.phase, session.alpha)
                    loss = -fake_predict
                    loss.backward()
                    session.optimizerG.step()
                    
                elif train_mode == config.MODE_CYCLIC:
                    kls = decoder_train(session, batch_size(reso), stats, kls)

            accumulate(g_running, generator)

            del real_image

            if train_mode == config.MODE_CYCLIC:
                if args.use_TB:
                    for key,val in stats.items():                    
                        writer.add_scalar(key, val, session.sample_i)
                elif batch_count % 100 == 0:
                    print(stats)

            if args.use_TB:
                writer.add_scalar('LOD', session.phase + session.alpha, session.sample_i)

        ########################  Statistics ######################## 

        b = batch_size_by_phase(session.phase)
        zr, xr = (stats['z_reconstruction_error'], stats['x_reconstruction_error']) if train_mode == config.MODE_CYCLIC else (0.0, 0.0)
        e = (session.sample_i / float(epoch_len))
        pbar.set_description(
            ('{0}; it: {1}; phase: {2}; b: {3:.1f}; Alpha: {4:.3f}; Reso: {5}; E: {6:.2f}; KL(real/fake/fakeG): {7}; z-reco: {8:.2f}; x-reco {9:.3f}; real_var {10:.4f}').format(batch_count+1, session.sample_i+1, session.phase, b, session.alpha, reso, e, kls, zr, xr, stats['real_var'])
            )
            #(f'{i + 1}; it: {iteration+1}; b: {b:.1f}; G: {gen_loss_val:.5f}; D: {disc_loss_val:.5f};'
            # f' Grad: {grad_loss_val:.5f}; Alpha: {alpha:.3f}; Reso: {reso}; S-mean: {real_mean:.3f}; KL(real/fake/fakeG): {kls}; z-reco: {zr:.2f}'))

        pbar.update(batch_size(reso))
        session.sample_i += batch_size(reso) # if not benchmarking else 100
        batch_count += 1

        ########################  Saving ######################## 

        if (batch_count % args.checkpoint_cycle == 0) or session.sample_i >= total_steps:
            for postfix in {'latest', str(session.sample_i).zfill(6)}:
                session.save_all('{}/{}_state'.format(args.checkpoint_dir, postfix))
                saveSNU('{}/{}_SNU'.format(args.checkpoint_dir, postfix))

            print("Checkpointed to {}".format(session.sample_i))

        ########################  Tests ######################## 

        try:
            evaluate.tests_run(g_running, encoder, test_data_loader, session, writer,
            reconstruction       = (batch_count % 800 == 0),
            interpolation        = (batch_count % 800 == 0),
            collated_sampling    = (batch_count % 800 == 0),
            individual_sampling  = (batch_count % (args.images_per_stage/batch_size(reso)/4) == 0)
            )
        except (OSError, StopIteration):
            print("Skipped periodic tests due to an exception.")

    pbar.close()

def makeTS(opts, session):
    print("Using preset training scheduler of celebaHQ and LSUN Bedrooms to set phase length, LR and KL margin. To override this behavior, modify the makeTS() function.")

    ts = TrainingScheduler(opts, session)

    if args.data == 'celebaHQ':
# Enabling a wide margin also in the beginning of the training sometimes helps to prevent early training collapse.
# In such a scenario, uncommenting the following lines (and replacing the regular (0,4) range) may help.
#        for p in range(0,2):
#            ts.add(p*2400, _phase=p, _lr=[0.0005, 0.0005], _margin=0.20, _aux_operations=None)
#        for p in range(2,4):
        for p in range(0,4):
            ts.add(p*2400, _phase=p, _lr=[0.0005, 0.0005], _margin=0.05, _aux_operations=None)

    if args.data == 'celebaHQ':
        ts.add(9600, _phase=4, _lr=[0.001, 0.001], _margin=0, _aux_operations=None) #0.02
        ts.add(13000, _phase=4, _lr=[0.001, 0.001], _margin=0.02, _aux_operations=None) #9600...16520 = 6920
        ts.add(16520, _phase=5, _lr=[0.001, 0.001], _margin=0.02, _aux_operations=None) #16520...20040 = 3520
        ts.add(20040, _phase=6, _lr=[0.001, 0.001], _margin=0.04, _aux_operations=None)
    elif args.data == 'lsun':
        for p in range(4):
            ts.add(p*2400, _phase=p, _lr=[0.001, 0.001], _margin=0, _aux_operations=None)
        ts.add(9600, _phase=4, _lr=[0.001, 0.001], _margin=0, _aux_operations=None)
        ts.add(12360, _phase=4, _lr=[0.001, 0.001], _margin=0.06, _aux_operations=None)
        ts.add(15400, _phase=5, _lr=[0.001, 0.001], _margin=0.06, _aux_operations=None)
        ts.add(21400, _phase=6, _lr=[0.001, 0.001], _margin=0.04, _aux_operations=None)
    elif args.data != 'cifar10':
        for p in range(4):
            ts.add(p*2400, _phase=p, _lr=[0.001, 0.001], _margin=0, _aux_operations=None)
        for p in range(4,8):
            ts.add(p*2400, _phase=p, _lr=[0.001, 0.001], _margin=0, _aux_operations=None)

    return ts

def main():
    setup()
    session = Session()
    session.create()

    print('PyTorch {}'.format(torch.__version__))

    if args.train_path:
        train_data_loader = data.get_loader(args.data, args.train_path)
    else:
        train_data_loader = None

    if args.test_path or args.z_inpath:
        test_data_loader = data.get_loader(args.data, args.test_path)
    elif args.aux_inpath:
        test_data_loader = data.get_loader(args.data, args.aux_inpath)
    else:
        test_data_loader = None

    # 4 modes: Train (with data/train), test (with data/test), aux-test (with custom aux_inpath), dump-training-set

    if args.run_mode == config.RUN_TRAIN:
        scheduler = makeTS([session.optimizerD, session.optimizerG], session)
        train(session.generator, session.encoder, session.g_running, train_data_loader, test_data_loader,
            session  = session,
            total_steps = args.total_kimg * 1000,
            train_mode = args.train_mode,
            sched = scheduler)
    elif args.run_mode == config.RUN_TEST:
#        evaluate.Utils.reconstruction_dryrun(session.g_running, session.encoder, test_data_loader, session=session)
        # Enable dry_run if SNU of the checkpoint is not available
        evaluate.tests_run(session.g_running, session.encoder, test_data_loader, session=session, writer=writer)
    elif args.run_mode == config.RUN_DUMP:
        session.phase = args.start_phase
        data.dump_training_set(train_data_loader, args.dump_trainingset_N, args.dump_trainingset_dir, session)

if __name__ == '__main__':
    main()
