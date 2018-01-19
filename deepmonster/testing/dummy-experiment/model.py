from deepmonster.machinery.model import Model, graph_defining_method
import theano
import theano.tensor as T

class DummyModel(Model):
    def build_fprop_graph(self):
        self.some_graph1()
        self.some_graph2()

    @graph_defining_method('y', 'graph1')
    def some_graph1(self):
        x = T.fmatrix('inputs')
        y = self.architecture['fully'].fprop(x)
        ysum = y.sum()
        ysum.name = 'ysum'
        self.track_var(ysum)
        return locals()

    @graph_defining_method('z', 'graph2')
    def some_graph2(self):
        exec self.fetch_and_assign_by_exec('y', 'graph1')
        z = y**2
        z.name = 'z'
        self.track_var(z)
