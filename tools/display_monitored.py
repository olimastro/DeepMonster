import cPickle as pkl
import matplotlib.pylab as plt

from collections import OrderedDict


def display(pklfile, request=['train_data_accuracy', 'train_sample_accuracy']) :
    for key in request :
        plt.plot(pklfile[key], label=key)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=len(request), mode="expand", borderaxespad=0.)
    plt.show()


def parse_requests(path) :
    #instance of OrderedDict
    odict = OrderedDict(pkl.load(open(path,'r')))
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


if __name__ == '__main__' :
    import sys
    if sys.argv[1] == '--all' :
        odict = pkl.load(open(sys.argv[2],'r'))
        display(odict, odict.keys())

    else :
        parse_requests(sys.argv[1])
