import argparse
import time
import sys, os
import numpy as np
import matplotlib.pylab as plt
import PIL.Image as Image

from subprocess import call

def animate(y, ndim, cmap) :
    plt.ion()

    if ndim == 5:
        plt.figure()
        plt.show()
        for i in range(y.shape[1]) :
            print "Showing batch", i
            plt.close('all')
            for j in range(y.shape[0]) :
                plt.imshow(y[j,i], interpolation='none', cmap=cmap)
                plt.pause(0.1)

            time.sleep(1)
    else:
        for i in range(y.shape[1]) :
            print "Showing batch", i
            plt.close('all')
            for j in range(y.shape[0]) :
                plt.figure(0)
                plt.imshow(y[j,i], interpolation='none', cmap=cmap)
                plt.figure(1)
                plt.imshow(x[j,i], interpolation='none', cmap=cmap)
                plt.pause(0.2)

            time.sleep(1)


def show_samples(y, ndim, nb=10, cmap=''):
    if ndim == 4:
        for i in range(nb**2):
            plt.subplot(nb, nb, i+1)
            plt.imshow(y[i], cmap=cmap, interpolation='none')
            plt.axis('off')

    else:
        x = y[0]
        y = y[1]
        plt.figure(0)
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.imshow(x[i], cmap=cmap, interpolation='none')
            plt.axis('off')

        plt.figure(1)
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.imshow(y[i], cmap=cmap, interpolation='none')
            plt.axis('off')

    plt.show()


def fancy_show(y, cmap=''):
    x = y[0]
    y = y[1]

    plt.figure(0)
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.imshow(x[i], cmap=cmap, interpolation='none')
        plt.axis('off')
    plt.figure(1)
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.imshow(y[i], cmap=cmap, interpolation='none')
        plt.axis('off')
    plt.show()



def create_movie(movie_npz):
    if movie_npz.shape[3] % 2 != 0:
        print "ffmpeg wants even pixels, cropping a tinsy bit"
        _movie_npz = np.empty(movie_npz.shape[:3]+(movie_npz.shape[3]-1,movie_npz.shape[4]-1))
        _movie_npz = movie_npz[:,:,:,:_movie_npz.shape[3],:_movie_npz.shape[4]]
        movie_npz = _movie_npz
    movie_npz = movie_npz.transpose(0,1,3,4,2)
    if movie_npz.shape[-1] == 1 :
        new = np.ones(movie_npz.shape[:-1]+(3,))
        new[:,:,:,:,0] = movie_npz[:,:,:,:,0]
        new[:,:,:,:,1] = movie_npz[:,:,:,:,0]
        new[:,:,:,:,2] = movie_npz[:,:,:,:,0]
        movie_npz = new
    elif movie_npz.shape[-1] != 3 :
        raise ValueError("Shape of channels should be 1 or 3")
    movie_npz = (movie_npz*255).astype('uint8')

    local_dir = os.environ['TMPDIR']+"/.tmp/"
    call(['mkdir','--parents',local_dir])

    k = 0 ; J = movie_npz.shape[0]
    for i in range(movie_npz.shape[1]):
        for j in range(movie_npz.shape[0]):
            im = Image.fromarray(movie_npz[j,i,:,:,:])
            im.save(local_dir+'temp_'+str(i*J+j+k+1)+'.png')
        k += 1
        transition_image = np.zeros(movie_npz.shape[2:]).astype('uint8')
        im = Image.fromarray(transition_image)
        j = movie_npz.shape[0] - 1
        im.save(local_dir+'temp_'+str(i*J+j+k+1)+'.png')

    call(['ffmpeg', '-framerate', '5', '-i', local_dir+'temp_%d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', os.environ['HOME']+'/samples.mp4'])
    call(['rm', '-r', local_dir])
    sys.exit()


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    action_type = parser.add_mutually_exclusive_group()
    action_type.add_argument('--ffmpeg', action='store_true',
                             help="Create a mp4 out of samples")
    action_type.add_argument('-s', '--sample', action='store_true',
                             help="Show or animate a set of samples")
    action_type.add_argument('-r', '--reconstruct', action='store_true',
                             help="Show or animate a set of reconstructions")
    action_type.add_argument('--fancy', action='store_true',
                             help="Show fancy reconstructions")
    parser.add_argument('file', metavar='file', type=str,
                        help="Path to file")
    args = parser.parse_args()

    # tbc01
    # they have been saved with savez
    y = np.load(args.file)
    y = y[y.keys()[0]]
    # this if is to try to save a mistake into a th.function
    if y.ndim == 6 and (arg == '-cm' or arg == '-s'):
        y = y[0]
    if np.min(y) < -0.9 :
        print "renormalizing from 0 to 1"
        # probably these were normalized -1 to 1
        y += 1.
        y /= 2.
    print y.shape


    if args.ffmpeg:
        create_movie(y)

    pattern = range(y.ndim)
    pattern[-3:] = [pattern[-2],pattern[-1],pattern[-3]]
    y = y.transpose(*pattern)
    if y.shape[-1] == 1:
        y = y[...,0]
        cmap = 'gray'
        ndim = y.ndim + 1
    else:
        cmap = 'Spectral'
        ndim = y.ndim


    if args.sample:
        if ndim == 4 :
            show_samples(y, ndim, cmap=cmap)
        elif ndim == 5 :
            animate(y, ndim, cmap=cmap)
    elif args.reconstruct:
        if ndim == 5 :
            show_samples(y, ndim, cmap=cmap)
        elif ndim == 6 :
            animate(y, ndim, cmap=cmap)
    elif args.fancy:
        fancy_show(y, cmap)
