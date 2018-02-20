import argparse
import cPickle as pkl
import matplotlib.pylab as plt
import numpy as np

from collections import OrderedDict


def display(pklfile, request=['train_data_accuracy', 'train_sample_accuracy']) :
    for key in request :
        plt.plot(pklfile[key], label=key)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=len(request), mode="expand", borderaxespad=0.)
    plt.show()


def parse_requests(odict) :
    #instance of OrderedDict
    display_dict = zip(range(len(odict.keys())), odict.keys())
    print
    print "There are the following keys in the pkl :"
    #print odict.keys()
    #print range(len(odict.keys()))
    print display_dict
    print
    figures = int(raw_input("How many figures would you like? >>> "))
    request = OrderedDict()
    print
    for i in range(figures) :
        print "For figure #"+str(i)+" what would you plot?"
        thisrequest = raw_input("Type in the order as they appear the keys in the pkl file seperated by space >>> ")
        thisrequest = thisrequest.split(' ')
        request_as_listofstr = []
        for j in thisrequest :
            request_as_listofstr += [display_dict[int(j)][1]]
        request.update({i : request_as_listofstr})
        print

    for i in range(figures) :
        plt.figure(i)
        display(odict, request[i])


def parse_ml_log(path):
    # the main loop log has as keys the iteration index, we want to aggregate
    # by monitored quantity instead
    odict = {}
    with open(path, 'r') as f:
        log = pkl.load(f)

    for k, v in log.iteritems():
        if not isinstance(v, dict):
            continue
        for kv, vv in v.iteritems():
            if not odict.has_key(kv):
                odict.update({kv: []})
            odict[kv].append(vv)

    for k, v in odict.iteritems():
        odict[k] = np.stack(v)

    return odict


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    action_type = parser.add_mutually_exclusive_group()
    action_type.add_argument('--all', action='store_true',
                             help="Display all entries")
    action_type.add_argument('--no-mll-parse', action='store_true',
                         help="Don't preparse the pkl file as a mainloop log")
    parser.add_argument('path', metavar='path', type=str,
                        help="Path to file")
    args = parser.parse_args()
    path = args.path

    if args.no_mll_parse:
        with open(path, 'r') as f:
            odict = pkl.load(f)
    else:
        odict = parse_ml_log(path)

    if args.all:
        display(odict, odict.keys())
    else :
        parse_requests(odict)
