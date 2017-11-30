import os
import argparse
from os_call_check import OsC
osc = OsC(False)

def mode1():
    # use this mode to give a folder and process all npz in it
    # depending on their samples nb tag, will only process one name
    # which it takes from the folder name
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='file', type=str)
    parser.add_argument("-t", "--tar", action='store_true')
    range_type = parser.add_mutually_exclusive_group(required=True)
    range_type.add_argument("-l", "--list", nargs='+')
    range_type.add_argument("-r", "--range", nargs=2)
    args = parser.parse_args()

    path = args.file[:-1] if args.file[-1] == '/' else args.file
    name = path.split('/')[-1]

    if args.list is not None:
        requests = [int(x) for x in args.list]
        assert len(set(requests)) == len(requests), "should be all unique int"
    else:
        requests = range(0, int(args.range[0])+1, int(args.range[1]))

    try:
        tmpdir = os.environ['TMPDIR']
    except KeyError:
        tmpdir = '/tmp/'
    tmpsample = os.path.join(os.environ['HOME'], 'tmpsample')
    osc.call('mkdir --parents ' + tmpsample)

    for i in requests:
        osc.call('TMPDIR={} python {}/plot.py --ffmpeg {}'.format(
            tmpdir, os.environ['PWD'],
            os.path.join(path, name + '_samples' + str(i) + '.npz')))
        osc.call('mv {} {}'.format(
            os.path.join(os.environ['HOME'], 'samples.mp4'),
            os.path.join(tmpsample, name + '_' + str(i) + '.mp4')))

    if args.tar:
        osc.call('tar -cvf {}.tar -C {} .'.format(
            os.path.join(os.environ['HOME'], name), tmpsample))
    else:
        osc.call('mv {}/* {}/'.format(tmpsample, os.environ['HOME']))

    osc.call('rm -r ' + tmpsample)


def mode2():
    # use this mode to give a folder and process all npz in it, regardless
    # of their names. Optionnally only those with a keyword in them.
    # Will skip the non npz files.
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='file', type=str)
    parser.add_argument("-k", "--key", type=str, default='')
    parser.add_argument("-t", "--tar", action='store_true')
    args = parser.parse_args()

    path = args.file[:-1] if args.file[-1] == '/' else args.file
    keyword = None if args.key == '' else args.key

    try:
        tmpdir = os.environ['TMPDIR']
    except KeyError:
        tmpdir = '/tmp/'
    tmpsample = os.path.join(os.environ['HOME'], 'tmpsample')
    osc.call('mkdir --parents ' + tmpsample)

    for npz in os.listdir(path):
        name = npz[:-4]
        if keyword is not None and not keyword in npz:
            continue
        if npz[-4:] != '.npz':
            continue

        osc.call('TMPDIR={} python {}/plot.py --ffmpeg {}'.format(
            tmpdir, os.environ['PWD'],
            os.path.join(path, npz)))
        osc.call('mv {} {}'.format(
            os.path.join(os.environ['HOME'], 'samples.mp4'),
            os.path.join(tmpsample, name + '.mp4')))

    if args.tar:
        osc.call('tar -cvf {}.tar -C {} .'.format(
            os.path.join(os.environ['HOME'], 'samples'), tmpsample))
    else:
        osc.call('mv {}/* {}/'.format(tmpsample, os.environ['HOME']))

    osc.call('rm -r ' + tmpsample)


if __name__ == '__main__':
    i = int(raw_input("mode1 or 2? (type an int or crash)"))
    if i == 1:
        mode1()
    elif i == 2:
        mode2()
