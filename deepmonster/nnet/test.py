import numpy as np
import theano
import theano.tensor as T


class Test(object):
    """
        This class is meant to take Feedforward objects, build theano
        function with it, try to run it and print the output's shape
    """
    def __init__(self, dict_of_test, mode='crash'):
        """
            dict_of_test is :
                {Feedforward : [input_shape]}
        """
        self.dict_of_test = dict_of_test
        self.mode = mode


    def run(self):
        print "Starting tests..."
        print
        for feedforward, test_info in self.dict_of_test.iteritems():
            if len(test_info[0]) == 5:
                dtensor5 = T.TensorType('float32', (False,)*5)
                x = dtensor5('x')
            elif len(test_info[0]) == 4:
                x = T.ftensor4('x')
            elif len(test_info[0]) == 3:
                x = T.ftensor3('x')
            elif len(test_info[0]) == 2:
                x = T.fmatrix('x')

            print "Testing " + feedforward.prefix

            out = feedforward.fprop(x)
            f = theano.function([x], out)
            npx = np.random.random(test_info[0]).astype(np.float32)
            if self.mode is 'no_crash' :
                try:
                    out_shape = f(npx).shape
                    print out_shape
                except:
                    print "Error encountered in this network"
            else :
                out_shape = f(npx).shape
                print out_shape

            print
        print "Finished"
        print
