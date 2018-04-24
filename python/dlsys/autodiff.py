"""A library to take autodiff and execute a computation graph """
from __future__ import absolute_import

import numpy as np
import tvm
from . import tvm_op

class Node(object):
    """Node in a computation graph."""
    def __init__(self):
        """Constructor, new node is indirectly created by Op object call method.

            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object,
                e.g. add_op if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant.
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = add_byconst_op(self, other)
        return new_node

    def __mul__(self, other):
        """Multiplying two nodes return a new node."""
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            # Mul by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = mul_byconst_op(self, other)
        return new_node

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow print to display node name."""
        return self.name


def Variable(name):
    """User defined variables in an expression.
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder_op()
    placeholder_node.name = name
    return placeholder_node


class Op(object):
    """Op represents operations performed on nodes."""
    def __call__(self):
        """Create a new node and associate the op object with the node.

        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.
        output_val: output value of the node, modified in-place.
        compiled_func: compiled function that can be called on function inputs
        """
        raise NotImplementedError

    def gradient(self, node, output_grad):
        """Given output gradient, compute partial gradient to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        """Given shapes of input nodes, compute shape of output node.

        Implementation note:
        It's simpler to treat shape of constants as (1,), so that constants can
        be stored as a numpy array too and you would need fewer special case
        handling.

        Parameters
        ----------
        node: node whose shape is being inferred.
        input_vals: shapes of input nodes.

        Returns
        -------
        A tuple representing the shape of output node.
        """
        raise NotImplementedError

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        """Compile the tvm function to native code for given input shapes.

        Parameters
        ----------
        node: node where the compute is done.
        input_shapes: shapes of input nodes.
        tgt: target device where computation is done, e.g. "llvm", "cuda", "arm" 
        tgt_host: target host where driver code is generated, e.g. "llvm"
               
        Returns
        -------
        A python function that you can directly call on op inputs and output.
        """
        raise NotImplementedError


class AddOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 2
        assert input_vals[0].shape == input_vals[1].shape
        compiled_func(input_vals[0], input_vals[1], output_val)  

    def gradient(self, node, output_grad):
        return [output_grad, output_grad]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        return broadcast_rule(input_shapes[0], input_shapes[1])

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_elemwise_add(
            input_shapes[0], tgt, tgt_host, "elem_add")


class AddByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 1
        compiled_func(input_vals[0], output_val)  

    def gradient(self, node, output_grad):
        return [output_grad]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        """TODO: Your code here"""

class MulOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 2
        assert input_vals[0].shape == input_vals[1].shape
        compiled_func(input_vals[0], input_vals[1], output_val)  

    def gradient(self, node, output_grad):
        return [node.inputs[1] * output_grad, node.inputs[0] * output_grad]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        """TODO: Your code here"""

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        """TODO: Your code here"""

class MulByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 1
        compiled_func(input_vals[0], output_val)  

    def gradient(self, node, output_grad):
        return [node.const_attr * output_grad]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        """TODO: Your code here"""

class MatMulOp(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (
            node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        compiled_func(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):
        if ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is False)):
            # if Y=AB, then dA=dY B^T, dB=A^T dY
            lhs_grad = matmul_op(
                output_grad, node.inputs[1], trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                node.inputs[0], output_grad, trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is False)):
            # if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A^T dY
            lhs_grad = matmul_op(
                node.inputs[1], output_grad, trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                node.inputs[0], output_grad, trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is True)):
            # if Y=A B^T, then dA=dY B^T, dB=(A^T dY)^T=dY^T A
            lhs_grad = matmul_op(
                output_grad, node.inputs[1], trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                output_grad, node.inputs[0], trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is True)):
            # if Y=A^T B^T, then dA=(dY B^T)^T=B dY^T, dB=(A^T dY)^T=dY^T A
            lhs_grad = matmul_op(
                node.inputs[1], output_grad, trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                output_grad, node.inputs[0], trans_A=True, trans_B=False)
        return [lhs_grad, rhs_grad]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        """TODO: Your code here"""
        

class PlaceholderOp(Op):
    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert False, "placeholder %s values provided by feed_dict" % node.name

    def gradient(self, node, output_grad):
        return None

    def infer_shape(self, node, input_shapes):
        assert False, "placeholder %s shape provided by feed_shape" % node.name

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return None

class ZerosLikeOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.zeros(node_A.shape)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 1
        output_val.copyfrom(
            np.zeros(input_vals[0].shape, dtype = input_vals[0].dtype))

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """If input_shape is a vector, simpler to return (1,)"""
        """TODO: Your code here"""

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return None


class OnesLikeOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.ones(node_A.shape)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 1
        output_val.copyfrom(
            np.ones(input_vals[0].shape, dtype = input_vals[0].dtype))

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """If input_shape is a vector, simpler to return (1,)"""
        """TODO: Your code here"""

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return None


class ReduceSumAxisZeroOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.sum(node_A, axis=0).
        Only support common-case axis=0 reduction for simplicity of gradient.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "ReduceSumAxisZero(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 1
        compiled_func(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        return [broadcastto_op(output_grad, node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """summation reduction axis = 0
        e.g. (3,4,5)->(4,5)
        for vector, simpler to do (3,)->(1,)
        """
        assert(len(input_shapes)==1)
        if len(input_shapes[0]) == 1:
            return (1,)
        return input_shapes[0][1:]

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_reduce_sum_axis_zero(
            input_shapes[0], tgt, tgt_host, "reduce_sum_axis_zero")


class BroadcastToOp(Op):
    def __call__(self, node_A, node_B):
        """Creates a node that represents np.broadcast_to(node_A, node_B.shape).
        Only support axis=0. e.g. (3,4)->(2,3,4) to make gradient simple.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "BroadcastTo(%s,%s.shape)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert(len(input_vals)==2)
        compiled_func(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        grad_A = reducesumaxiszero_op(output_grad)
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        """TODO: Your code here"""

def softmax_func(y):
    """Numerically stable softmax."""
    b = y - np.max(y, axis=1, keepdims=True)
    expb = np.exp(b)
    softmax = expb / np.sum(expb, axis=1, keepdims=True)
    return softmax


class SoftmaxCrossEntropyOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "SoftmaxXEntropy(%s,%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 2
        y = input_vals[0]
        y_ = input_vals[1]
        compiled_func(y, y_, output_val)

    def gradient(self, node, output_grad):
        grad_A_temp = softmax_op(node.inputs[0]) + -1 * node.inputs[1]
        grad_A = grad_A_temp * broadcastto_op(output_grad, grad_A_temp)
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        """TODO: Your code here"""

class SoftmaxOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Softmax(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 1
        compiled_func(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        # Do not directly use SoftmaxOp, use SoftmaxCrossEntropyOp instead.
        # Not allowing taking 2nd derivative of SoftmaxCrossEntropyOp.
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        """TODO: Your code here"""


class ReluOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Relu(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        compiled_func(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        return [relu_gradient_op(node.inputs[0], output_grad)]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        """TODO: Your code here"""


class ReluGradientOp(Op):
    def __call__(self, node_A, node_B):
        """node_B is output_grad"""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "ReluGradient(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        compiled_func(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        """TODO: Your code here"""

# Create global singletons of operators.
add_op = AddOp()
mul_op = MulOp()
add_byconst_op = AddByConstOp()
mul_byconst_op = MulByConstOp()
matmul_op = MatMulOp()
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
reducesumaxiszero_op = ReduceSumAxisZeroOp()
broadcastto_op = BroadcastToOp()
softmaxcrossentropy_op = SoftmaxCrossEntropyOp()
softmax_op = SoftmaxOp()
relu_op = ReluOp()
relu_gradient_op = ReluGradientOp()


class Executor(object):
    """Executor computes values for given set of nodes in computation graph."""
    def __init__(self, eval_node_list, ctx=None):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        ctx: runtime DLContext, default is None which means np.ndarray on cpu
        topo_order: list of nodes in topological order
        node_to_shape_map: dict from node to shape of the node
        node_to_arr_map: dict from node to tvm.nd.array allocated for node
        node_to_compiled_func: dict from node to compiled func for node
        feed_shapes: shapes of feed_dict from last run(...)
        """
        self.eval_node_list = eval_node_list
        self.ctx = ctx
        if self.ctx == tvm.cpu(0):
            self.tgt = "llvm"
            self.tgt_host="llvm"
        else:
            assert False, "non-CPU context not yet supported"
        self.topo_order = find_topo_sort(self.eval_node_list)
        self.node_to_shape_map = None
        self.node_to_arr_map = None
        self.node_to_compiled_func = None
        self.feed_shapes = None

    def infer_shape(self, feed_shapes):
        """Given shapes of feed_dict nodes, infer shape for all nodes in graph.

        Implementation note:
        Iteratively calls node.op.infer_shape to infer shapes.
        Node shapes stored in self.node_to_shape_map.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        """TODO: Your code here"""

    def memory_plan(self, feed_shapes):
        """Allocates tvm.nd.array for every node except feed_dict nodes.

        Implementation note:
        Easy Option: Alloc a tvm.nd.array per node that persists across run()

        Use self.node_to_arr_map to store node->tvm.nd.array mapping
        to allow mapping to persist across multiple executor.run().

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        """TODO: Your code here"""

    def compile_funcs(self, feed_shapes):
        """Compile tvm ops to native code.

        Must be called after infer_shape(...) since op compilation requires
        knowledge of tensor shapes.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        """TODO: Your code here"""

    def run(self, feed_dict, convert_to_numpy_ret_vals=False):
        """
        Parameters
        ----------
        feed_dict: a dictionary of node->np.ndarray supplied by user.
        convert_to_numpy_ret_vals: whether to convert ret vals to np.array.

        Returns
        -------
        A list of values for nodes in eval_node_list. tvm.nd.array or np.ndarray.
        """
        def are_feed_shapes_equal(sa, sb):
            if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
                return False
            unmatched_item = set(sa.items()) ^ set(sb.items())
            return len(unmatched_item) == 0

        node_to_val_map = {}
        for node, value in feed_dict.items():
            assert isinstance(value, tvm.ndarray.NDArray),\
                "feed_dict value type not supported"    
            node_to_val_map[node] = value

        # collect shapes for all placeholders
        feed_shapes = {}
        for node in node_to_val_map:
            feed_shapes[node] = node_to_val_map[node].shape

        # infer shape if feed_shapes changed since last run
        # e.g. call run() on test data after trainng
        if (not are_feed_shapes_equal(feed_shapes, self.feed_shapes)):
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes
            self.memory_plan(feed_shapes)
            self.compile_funcs(feed_shapes)

        # Traverse graph in topo order and compute values for all nodes.
        for node in self.topo_order:
            if node in node_to_val_map:
                # Skip placeholder nodes. Values already provided by feed_dict.
                continue
            input_vals = [node_to_val_map[n] for n in node.inputs]
            node_val = self.node_to_arr_map[node]
            # node_val is modified in-place
            node.op.compute(
                node, input_vals, node_val, self.node_to_compiled_func[node])
            node_to_val_map[node] = node_val
        # Collect node values.
        if convert_to_numpy_ret_vals:
            return [node_to_val_map[n].asnumpy() for n in self.eval_node_list]
        return [node_to_val_map[n] for n in self.eval_node_list]


def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    node_to_output_grad = {}
    # Traverse forward graph in reverse topological order
    reverse_topo_order = reversed(find_topo_sort([output_node]))
    for node in reverse_topo_order:
        output_grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = output_grad
        input_grads_list = node.op.gradient(node, output_grad)
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            # Calculate partial adjoint for input nodes.
            node_to_output_grads_list[node.inputs[i]].append(
                input_grads_list[i])

    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list

##################
# Helper Methods #
##################


def find_topo_sort(node_list):
    """Given a list of nodes, return a topo ordering of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a
    topological sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    """Custom sum func to avoid creating redundant nodes in Python sum func."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)


def broadcast_rule(shape_a, shape_b):
    """Return output shape of broadcast shape_a, shape_b.
    e.g. broadcast_rule((3,2), (4,3,2))
    returns output_shape = (4,3,2)

    Check out explanations and more examples at
    https://docs.scipy.org/doc/numpy-1.10.0/user/basics.broadcasting.html
    http://eli.thegreenplace.net/2015/broadcasting-arrays-in-numpy/
    """
    assert(isinstance(shape_a, tuple))
    assert(isinstance(shape_b, tuple))
    if len(shape_a) > len(shape_b):
        longer_shape, shorter_shape = shape_a, shape_b
    else:
        longer_shape, shorter_shape = shape_b, shape_a
    len_diff = len(longer_shape) - len(shorter_shape)
    for i in xrange(len_diff):
        # pad with leading 1s
        shorter_shape = (1,) + shorter_shape
    assert len(shorter_shape) == len(longer_shape)
    output_shape = list(longer_shape)
    for i in xrange(len(output_shape)):
        assert (shorter_shape[i] == longer_shape[i]) \
            or (shorter_shape[i] == 1) \
            or (longer_shape[i] == 1)
        output_shape[i] = max(shorter_shape[i], longer_shape[i])
    return tuple(output_shape)
