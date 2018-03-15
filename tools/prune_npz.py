import argparse
import os

from deepmonster.utils import sort_by_numbers_in_file_name

def prune_npz(path, limit=None):
    files = os.listdir(path)
    files = filter(lambda x: x[-4:] == '.npz', files)
    total_npz = len(files)

    # keep first, keep last
    files = files[1:-1]
    # remove half
    files = files[::2]

    if len(files) == 0:
        print "no npz"
        return 2
    elif limit is not None and total_npz - len(files) < limit:
        print "limit reached, won't delete more npz"
        return 1

    files = sort_by_numbers_in_file_name(files)
    for f in files:
        print "rm:", os.path.join(path, f)
        os.remove(os.path.join(path, f))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=int, default=1,
                        help="Ratio by which it halves the amount of npz "+\
                        " (2 = 1/4 of the amount is left). Special value: -1 == until limit is reached")
    parser.add_argument('--limit', type=int, default=10,
                        help="Minimum limit of npz to keep")
    parser.add_argument('-r', '--recursive', action='store_true',
                        help="Apply prune recursively in dir (only go at depth 1)")
    parser.add_argument('dir', metavar='dir', type=str,
                        help="Path to dir")
    args = parser.parse_args()

    dirpath = args.dir
    if args.recursive:
        d = os.listdir(dirpath)
        d = map(lambda x: os.path.join(dirpath, x), d)
        dirpath = filter(lambda x: os.path.isdir(x), d)
    else:
        dirpath = [dirpath]

    for path in dirpath:
        if args.ratio > 0:
            for i in range(args.ratio):
                prune_npz(path, args.limit)
        else:
            assert args.ratio == -1
            while True:
                rval = prune_npz(path, args.limit)
                if rval in [1,2]:
                    break
