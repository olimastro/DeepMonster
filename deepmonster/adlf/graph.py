import theano

# traverse the graph in search of a tag
def graph_traversal(x, attr):
    def one_step_deeper(L, var):
        if getattr(var.tag, attr, None) is not None:
            L += [var]
        owner = getattr(var, 'owner')
        if owner is None:
            return
        for v in owner.inputs:
            one_step_deeper(L, v)
        return

    rval = []
    one_step_deeper(rval, x)
    return rval


if __name__ == "__main__":
    import theano.tensor as T
    x = T.fmatrix('x')
    y = x + 2
    y.tag.bntag = 'kklol'
    z = y**3 - y + 4
    rval = graph_traversal(z, 'bntag')
    import ipdb; ipdb.set_trace()
    for v in rval:
        print v.tag.bntag
