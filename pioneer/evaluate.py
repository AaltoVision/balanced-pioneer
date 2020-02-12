from PIL import Image
import os
import numpy as np

import torch
from torch.autograd import Variable
import torchvision.utils

from pioneer import config
from pioneer import utils
from pioneer import data

#import bcolz

args = config.get_config()

class Utils:
    def reconstruction_dryrun(generator, encoder, loader, session):
        generator.eval()
        encoder.eval()

        utils.requires_grad(generator, False)
        utils.requires_grad(encoder, False)

        reso = 4 * 2 ** session.phase

        warmup_rounds = 200
        print('Warm-up rounds: {}'.format(warmup_rounds))

        if session.phase < 1:
            dataset = data.Utils.sample_data(loader, 4, reso)
        else:
            dataset = data.Utils.sample_data2(loader, 4, reso, session)
        real_image, _ = next(dataset)
        x = Variable(real_image).cuda()

        for i in range(warmup_rounds):
            ex = encoder(x, session.phase, session.alpha, args.use_ALQ).detach()
            ex, label = utils.split_labels_out_of_latent(ex)
            generator(ex, label, session.phase, session.alpha).detach()

        encoder.train()
        generator.train()

    #For loading eternally inputted latent codes, e.g. for Perceptual Path Length calculations
    @staticmethod
    def make(generator, session, writer):
        if not os.path.exists(args.x_outpath):
            os.makedirs(args.x_outpath)

        utils.requires_grad(generator, False)

        z = bcolz.carray(rootdir=args.z_inpath)
        print("Loading external z from {}, shape:".format(args.z_inpath))
        print(np.shape(z))
        N = np.shape(z)[0]

        samplesRepeatN =  1 + int(N / 8) #Assume large files...
        samplesDone = 0
        for outer_count in range(samplesRepeatN):
            samplesN = min(8, N - samplesDone)
            if samplesN <= 0:
                break
            zrange = z[samplesDone:samplesDone+samplesN, :]
            myz, input_class = utils.split_labels_out_of_latent(torch.from_numpy(zrange.astype(np.float32)))
            new_imgs = generator(
                myz,
                input_class,
                session.phase,
                session.alpha).detach().data.cpu()

            for ii, img in enumerate(new_imgs):
                torchvision.utils.save_image(
                    img,
                    '{}/{}.png'.format(args.x_outpath, str(ii + outer_count*8)), #.zfill(6)
                    nrow=args.n_label,
                    normalize=True,
                    range=(-1, 1),
                    padding=0)

            samplesDone += samplesN

    @staticmethod
    def generate_intermediate_samples(generator, global_i, session, writer=None, collateImages = True):
        generator.eval()
        with torch.no_grad():

            save_root = args.sample_dir if args.sample_dir != None else args.save_dir

            utils.requires_grad(generator, False)

            # Total number is samplesRepeatN * colN * rowN
            # e.g. for 51200 samples, outcome is 5*80*128. Please only give multiples of 128 here.
            samplesRepeatN = max(1, int(args.sample_N / 128)) if not collateImages else 1
            reso = 4 * 2 ** session.phase

            if not collateImages:
                special_dir = '{}/sample/{}'.format(save_root, str(global_i).zfill(6))
                while os.path.exists(special_dir):
                    special_dir += '_'

                os.makedirs(special_dir)

            for outer_count in range(samplesRepeatN):

                colN = 1 if not collateImages else min(10, int(np.ceil(args.sample_N / 4.0)))
                rowN = 128 if not collateImages else min(5, int(np.ceil(args.sample_N / 4.0)))
                images = []
                for _ in range(rowN):
                    myz = Variable(torch.randn(args.n_label * colN, args.nz)).cuda()
                    myz = utils.normalize(myz)
                    myz, input_class = utils.split_labels_out_of_latent(myz)

                    new_imgs = generator(
                        myz,
                        input_class,
                        session.phase,
                        session.alpha).detach().data.cpu()

                    images.append(new_imgs)

                if collateImages:
                    sample_dir = '{}/sample'.format(save_root)
                    if not os.path.exists(sample_dir):
                        os.makedirs(sample_dir)

                    save_path = '{}/{}.png'.format(sample_dir,str(global_i + 1).zfill(6))
                    torchvision.utils.save_image(
                        torch.cat(images, 0),
                        save_path,
                        nrow=args.n_label * colN,
                        normalize=True,
                        range=(-1, 1),
                        padding=0)
                    # Hacky but this is an easy way to rescale the images to nice big lego format:
                    im = Image.open(save_path)
                    im2 = im.resize((1024, 512 if reso < 256 else 1024))
                    im2.save(save_path)

                    #if writer:
                    #    writer.add_image('samples_latest_{}'.format(session.phase), torch.cat(images, 0), session.phase)
                else:
                    for ii, img in enumerate(images):
                        torchvision.utils.save_image(
                            img,
                            '{}/{}_{}.png'.format(special_dir, str(global_i + 1).zfill(6), ii+outer_count*128),
                            nrow=args.n_label * colN,
                            normalize=True,
                            range=(-1, 1),
                            padding=0)

        generator.train()

    reconstruction_set_x = None

    @staticmethod
    def reconstruct(input_image, encoder, generator, session):
        with torch.no_grad():
            ex = encoder(Variable(input_image), session.phase, session.alpha, args.use_ALQ).detach()
            ex, label = utils.split_labels_out_of_latent(ex)
            gex = generator(ex, label, session.phase, session.alpha).detach()            
        return gex.data[:]

    @staticmethod
    def reconstruct_images(generator, encoder, loader, global_i, nr_of_imgs, prefix, reals, reconstructions, session, writer=None): #of the form"/[dir]"
        generator.eval()
        encoder.eval()

        with torch.no_grad():
            utils.requires_grad(generator, False)
            utils.requires_grad(encoder, False)

            save_root = args.sample_dir if args.sample_dir != None else args.save_dir

            if reconstructions and nr_of_imgs > 0:
                reso = 4 * 2 ** session.phase

                # First, create the single grid

                if Utils.reconstruction_set_x is None or Utils.reconstruction_set_x.size(2) != reso or (session.phase >= 1 and session.alpha < 1.0):
                    if session.phase < 1:
                        dataset = data.Utils.sample_data(loader, min(nr_of_imgs, 16), reso)
                    else:
                        dataset = data.Utils.sample_data2(loader, min(nr_of_imgs, 16), reso, session)
                    Utils.reconstruction_set_x, _ = next(dataset)

                reco_image = Utils.reconstruct(Utils.reconstruction_set_x, encoder, generator, session)

                t = torch.FloatTensor(Utils.reconstruction_set_x.size(0) * 2, Utils.reconstruction_set_x.size(1),
                                    Utils.reconstruction_set_x.size(2), Utils.reconstruction_set_x.size(3))

                t[0::2] = Utils.reconstruction_set_x[:]
                t[1::2] = reco_image

                save_path = '{}{}/reconstructions_{}_{}_{}.png'.format(save_root, prefix, session.phase, global_i, session.alpha)
                grid = torchvision.utils.save_image(t[:nr_of_imgs] / 2 + 0.5, save_path, padding=0)

                # Hacky but this is an easy way to rescale the images to nice big lego format:
                if session.phase < 4:
                    h = np.ceil(nr_of_imgs / 8)
                    h_scale = min(1.0, h/8.0)
                    im = Image.open(save_path)
                    im2 = im.resize((1024, int(1024 * h_scale)))
                    im2.save(save_path)

                #if writer:
                #    writer.add_image('reconstruction_latest_{}'.format(session.phase), t[:nr_of_imgs] / 2 + 0.5, session.phase)

                # Second, create the Individual images:
                if session.phase < 1:
                    dataset = data.Utils.sample_data(loader, 1, reso)
                else:
                    dataset = data.Utils.sample_data2(loader, 1, reso, session)

                special_dir = '{}/{}'.format(save_root if not args.aux_outpath else args.aux_outpath, str(global_i).zfill(6))

                if not os.path.exists(special_dir):
                    os.makedirs(special_dir)                

                print("Save images: Alpha={}, phase={}, images={}, at {}".format(session.alpha, session.phase, nr_of_imgs, special_dir))
                for o in range(nr_of_imgs):
                    if o%500==0:
                        print(o)

                    real_image, _ = next(dataset)
                    reco_image = Utils.reconstruct(real_image, encoder, generator, session)

                    t = torch.FloatTensor(real_image.size(0) * 2, real_image.size(1),
                                        real_image.size(2), real_image.size(3))      

                    save_path_A = '{}/{}_orig.png'.format(special_dir, o)
                    save_path_B = '{}/{}_pine.png'.format(special_dir, o)

                    torchvision.utils.save_image(real_image[0] / 2 + 0.5, save_path_A, padding=0)
                    torchvision.utils.save_image(reco_image[0] / 2 + 0.5, save_path_B, padding=0)

        encoder.train()
        generator.train()

    @staticmethod
    def slerp(p0, p1, t):
        omega = np.arccos(np.dot(p0,p1) / np.sqrt(np.dot(p0,p0)) / np.sqrt(np.dot(p1,p1)))
        k1 = np.sin((1-t) * omega) / np.sin(omega)
        k2 = np.sin(t * omega) / np.sin(omega)
        return k1*p0 + k2*p1

    interpolation_set_x = None
    @staticmethod
    def interpolate_images(generator, encoder, loader, epoch, prefix, session, writer=None):
        generator.eval()
        encoder.eval()

        with torch.no_grad():
            utils.requires_grad(generator, False)
            utils.requires_grad(encoder, False)

            nr_of_imgs = 4 if not args.hexmode else 6 # "Corners"
            reso = 4 * 2 ** session.phase
            if True:
                if session.phase < 1:
                    dataset = data.Utils.sample_data(loader, nr_of_imgs, reso)
                else:
                    dataset = data.Utils.sample_data2(loader, nr_of_imgs, reso, session)
                real_image, _ = next(dataset)
                Utils.interpolation_set_x = Variable(real_image, volatile=True).cuda()

            latent_reso_hor = 8
            latent_reso_ver = 8

            x = Utils.interpolation_set_x

            if args.hexmode:
                x = x[:nr_of_imgs] #Corners
                z0 = encoder(Variable(x), session.phase, session.alpha, args.use_ALQ).detach()

                X = np.array([-1.7321,0.0000 ,1.7321 ,-2.5981,-0.8660,0.8660 ,2.5981 ,-3.4641,-1.7321,0.0000 ,1.7321 ,3.4641 ,-2.5981,-0.8660,0.8660 ,2.5981 ,-1.7321,0.0000,1.7321])
                Y = np.array([-3.0000,-3.0000,-3.0000,-1.5000,-1.5000,-1.5000,-1.5000,0.0000,0.0000,0.0000,0.0000,0.0000,1.5000,1.5000,1.5000,1.5000,3.0000,3.0000,3.0000])
                corner_indices = np.array([0, 2, 7, 11, 16, 18])
                inter_indices = [i for i in range(19) if not i in corner_indices]
                edge_indices = np.array([1,3,6,12,15,17])
                middle_index = 9
                ring_indices = np.array([4,5,8,9,10,13,14])

                #distances_to_corners = np.sqrt(np.power(X - X[corner_indices[0]], 2) + np.power(Y - Y[corner_indices[0]], 2))
                distances_to_corners = [np.sqrt(np.power(X - X[corner_indices[i]], 2) + np.power(Y - Y[corner_indices[i]], 2)) for i in range(6)]
                weights = 1.0 - distances_to_corners / np.max(distances_to_corners)
                z0_x = torch.zeros((19,args.nz+args.n_label)).cuda()
                z0_x[corner_indices,:] = z0
                # Linear:
                ###z0x[inter_indices,:] = (weights.T @ z0)[inter_indices,:]
                #z0_x[inter_indices,:] = (torch.from_numpy(weights.T.astype(np.float32)).cuda() @ z0)[inter_indices,:]

                z0_x[edge_indices[0],:] = Utils.slerp(z0[0], z0[1], 0.5)
                z0_x[edge_indices[1],:] = Utils.slerp(z0[0], z0[2], 0.5)
                z0_x[edge_indices[2],:] = Utils.slerp(z0[1], z0[3], 0.5)
                z0_x[edge_indices[3],:] = Utils.slerp(z0[2], z0[4], 0.5)
                z0_x[edge_indices[4],:] = Utils.slerp(z0[3], z0[5], 0.5)
                z0_x[edge_indices[5],:] = Utils.slerp(z0[4], z0[5], 0.5)

                db = 0.125
                if False:
                    ring_ws = np.array([[.250,.375, .375],
                                        [.625,.250, .375],
                                        [.375,.625, .250],
                                        [.500,.500, .500],
                                        [.625,.375, .750],
                                        [.625,.750, .375],
                                        [.750,.625, .625]])

                ring_ws = np.array([[.250-db,.375-db, .375-db],
                                    [.625+db,.250-db, .375-db],
                                    [.375-db,.625+db, .250-db],
                                    [.500,.500, .500],
                                    [.625+db,.375-db, .750+db],
                                    [.625+db,.750+db, .375-db],
                                    [.750+db,.625+db, .625+db]])
                for r in range(7):
                    mz05 = Utils.slerp(z0[0], z0[5], ring_ws[r][0])
                    mz14 = Utils.slerp(z0[1], z0[4], ring_ws[r][1])
                    mz23 = Utils.slerp(z0[2], z0[3], ring_ws[r][2])
                    mz0514 = Utils.slerp(mz05, mz14, 0.5)
                    mz = Utils.slerp(mz0514, mz23, 0.5)
                    z0_x[ring_indices[r],:] = mz

                z0_x = utils.normalize(z0_x)
            else:
                z0 = encoder(Variable(x), session.phase, session.alpha, args.use_ALQ).detach()

            t = torch.FloatTensor(latent_reso_hor * (latent_reso_ver+1) + nr_of_imgs, x.size(1),
                                x.size(2), x.size(3))
            t[0:nr_of_imgs] = x.data[:]

            save_root = args.sample_dir if args.sample_dir != None else args.save_dir
            special_dir = save_root if not args.aux_outpath else args.aux_outpath

            if not os.path.exists(special_dir):
                os.makedirs(special_dir)

            for o_i in range(nr_of_imgs):
                single_save_path = '{}{}/hex_interpolations_{}_{}_{}_orig_{}.png'.format(special_dir, prefix, session.phase, epoch, session.alpha, o_i)
                grid = torchvision.utils.save_image(x.data[o_i] / 2 + 0.5, single_save_path, nrow=1, padding=0) #, normalize=True) #range=(-1,1)) #, normalize=True) #, scale_each=True)?

            if args.hexmode:
                z0_x, label = utils.split_labels_out_of_latent(z0_x)
                gex = generator(z0_x, label, session.phase, session.alpha).detach()                

                for x_i in range(19):
                    single_save_path = '{}{}/hex_interpolations_{}_{}_{}x{}.png'.format(special_dir, prefix, session.phase, epoch, session.alpha, x_i)
                    grid = torchvision.utils.save_image(gex.data[x_i] / 2 + 0.5, single_save_path, nrow=1, padding=0) #, normalize=True) #range=(-1,1)) #, normalize=True) #, scale_each=True)?

                return

            # Origs on the first row here
            # Corners are: z0[0] ... z0[1]
            #                .   
            #                .
            #              z0[2] ... z0[3]                

            delta_z_ver0 = ((z0[2] - z0[0]) / (latent_reso_ver - 1))
            delta_z_verN = ((z0[3] - z0[1]) / (latent_reso_ver - 1))
            for y_i in range(latent_reso_ver):
                if False: #Linear interpolation
                    z0_x0 = z0[0] + y_i * delta_z_ver0
                    z0_xN = z0[1] + y_i * delta_z_verN
                    delta_z_hor = (z0_xN - z0_x0) / (latent_reso_hor - 1)
                    z0_x = Variable(torch.FloatTensor(latent_reso_hor, z0_x0.size(0)))

                    for x_i in range(latent_reso_hor):
                        z0_x[x_i] = z0_x0 + x_i * delta_z_hor

                if True: #Spherical
                    t_y = float(y_i) / (latent_reso_ver-1)
                    #z0_y = Variable(torch.FloatTensor(latent_reso_ver, z0.size(0)))
                    z0_y1 = Utils.slerp(z0[0].data, z0[2].data, t_y)
                    z0_y2 = Utils.slerp(z0[1].data, z0[3].data, t_y)
                    z0_x = Variable(torch.FloatTensor(latent_reso_hor, z0[0].size(0)))
                    for x_i in range(latent_reso_hor):
                        t_x = float(x_i) / (latent_reso_hor-1)
                        z0_x[x_i] = Utils.slerp(z0_y1, z0_y2, t_x)

                z0_x, label = utils.split_labels_out_of_latent(z0_x)
                gex = generator(z0_x, label, session.phase, session.alpha).detach()
                
                # Recall that yi=0 is the original's row:
                t[(y_i+1) * latent_reso_ver:(y_i+2)* latent_reso_ver] = gex.data[:]

                for x_i in range(latent_reso_hor):
                    single_save_path = '{}{}/interpolations_{}_{}_{}_{}x{}.png'.format(special_dir, prefix, session.phase, epoch, session.alpha, y_i, x_i)
                    grid = torchvision.utils.save_image(gex.data[x_i] / 2 + 0.5, single_save_path, nrow=1, padding=0) #, normalize=True) #range=(-1,1)) #, normalize=True) #, scale_each=True)?
            
            save_path = '{}{}/interpolations_{}_{}_{}.png'.format(special_dir, prefix, session.phase, epoch, session.alpha)
            grid = torchvision.utils.save_image(t / 2 + 0.5, save_path, nrow=latent_reso_ver, padding=0) #, normalize=True) #range=(-1,1)) #, normalize=True) #, scale_each=True)?
            # rescale the images to nice big lego format:
            if session.phase < 4:
                im = Image.open(save_path)
                im2 = im.resize((1024, 1024))
                im2.save(save_path)        

            #if writer:
            #    writer.add_image('interpolation_latest_{}'.format(session.phase), t / 2 + 0.5, session.phase)

        generator.train()
        encoder.train()

def tests_run(generator_for_testing, encoder, test_data_loader, session, writer, reconstruction = True, interpolation = True, collated_sampling = True, individual_sampling = True):
    if args.sample_dir and not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    if args.z_inpath:
        Utils.make(generator_for_testing, session=session, writer=writer)
    if reconstruction:
        Utils.reconstruct_images(generator_for_testing, encoder, test_data_loader, session.sample_i, nr_of_imgs = args.reconstructions_N, prefix = '', reals = False, reconstructions=True, session=session, writer=writer)
    if interpolation:
        for ii in range(args.interpolate_N):
            Utils.interpolate_images(generator_for_testing, encoder, test_data_loader, session.sample_i+ii, prefix = '',session=session,writer=writer)
    if collated_sampling and args.sample_N > 0:        
        Utils.generate_intermediate_samples(
            generator_for_testing,
            session.sample_i, session=session, writer=writer, collateImages = True) #True)
    if individual_sampling and args.sample_N > 0: #Full sample set generation.
        print("Full Test samples - generating...")
        Utils.generate_intermediate_samples(
            generator_for_testing,
            session.sample_i, session=session, collateImages = False)
        print("Full Test samples generated.")
