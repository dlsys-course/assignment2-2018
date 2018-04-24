import numpy as np
import tvm
from dlsys import autodiff, tvm_op

tgt_host="llvm"
tgt="llvm"
dtype = "float32"
ctx = tvm.context(tgt, 0)


def test_matrix_elementwise_add():
    shape = (500, 200)
    x = np.random.uniform(0, 10, size=shape).astype(dtype)
    y = np.random.uniform(0, 10, size=shape).astype(dtype)
    z = np.zeros(shape).astype(dtype)
    arr_x = tvm.nd.array(x, ctx=ctx)
    arr_y = tvm.nd.array(y, ctx=ctx)
    arr_z = tvm.nd.array(z, ctx=ctx)
    elemwise_add = tvm_op.make_elemwise_add(shape, tgt, tgt_host, "elem_add")
    elemwise_add(arr_x, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(x + y, z, rtol=1e-5)


def test_matrix_elementwise_add_by_const():
    shape = (2000, 3000)
    x = np.random.uniform(0, 10, size=shape).astype(dtype)
    const_val = np.random.uniform(0, 10)
    y = np.zeros(shape).astype(dtype)
    arr_x = tvm.nd.array(x, ctx=ctx)
    arr_y = tvm.nd.array(y, ctx=ctx)
    elemwise_add_by_const = tvm_op.make_elemwise_add_by_const(shape, const_val, tgt, tgt_host, "elem_add_by_const")
    elemwise_add_by_const(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(x + const_val, y, rtol=1e-5)


def test_matrix_elementwise_mul():
    shape = (500, 200)
    x = np.random.uniform(0, 10, size=shape).astype(dtype)
    y = np.random.uniform(0, 10, size=shape).astype(dtype)
    z = np.zeros(shape).astype(dtype)
    arr_x = tvm.nd.array(x, ctx=ctx)
    arr_y = tvm.nd.array(y, ctx=ctx)
    arr_z = tvm.nd.array(z, ctx=ctx)
    elemwise_mul = tvm_op.make_elemwise_mul(shape, tgt, tgt_host, "elem_add")
    elemwise_mul(arr_x, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(x * y, z, rtol=1e-5)


def test_matrix_elementwise_mul_by_const():
    shape = (2000, 3000)
    x = np.random.uniform(0, 10, size=shape).astype(dtype)
    const_val = np.random.uniform(0, 10)
    y = np.zeros(shape).astype(dtype)
    arr_x = tvm.nd.array(x, ctx=ctx)
    arr_y = tvm.nd.array(y, ctx=ctx)
    elemwise_mul_by_const = tvm_op.make_elemwise_mul_by_const(shape, const_val, tgt, tgt_host, "elem_mul_by_const")
    elemwise_mul_by_const(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(x * const_val, y, rtol=1e-5)


def test_matrix_multiply():
    shapeX = (500, 700)
    shapeY = (700, 1000)
    shapeZ = (500, 1000)
    x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
    y = np.random.uniform(0, 10, size=shapeY).astype(dtype)
    z = np.zeros(shapeZ).astype(dtype)
    arr_x = tvm.nd.array(x, ctx=ctx)
    arr_y = tvm.nd.array(y, ctx=ctx)
    arr_z = tvm.nd.array(z, ctx=ctx)
   
    matrix_mul = tvm_op.make_matrix_mul(shapeX, False, shapeY, False, tgt, tgt_host, "matrix_mul")
    matrix_mul(arr_x, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.dot(x, y), z, rtol=1e-5)

    shapeX = (1000, 500)
    shapeY = (2000, 500)
    shapeZ = (1000, 2000)
    x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
    y = np.random.uniform(0, 10, size=shapeY).astype(dtype)
    z = np.zeros(shapeZ).astype(dtype)
    arr_x = tvm.nd.array(x, ctx=ctx)
    arr_y = tvm.nd.array(y, ctx=ctx)
    arr_z = tvm.nd.array(z, ctx=ctx)

    matrix_mul = tvm_op.make_matrix_mul(shapeX, False, shapeY, True, tgt, tgt_host, "matrix_mul")
    matrix_mul(arr_x, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.dot(x, np.transpose(y)), z, rtol=1e-5)
    
    shapeX = (500, 1000)
    shapeY = (500, 2000)
    shapeZ = (1000, 2000)   
    x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
    y = np.random.uniform(0, 10, size=shapeY).astype(dtype)
    z = np.zeros(shapeZ).astype(dtype)
    arr_x = tvm.nd.array(x, ctx=ctx)
    arr_y = tvm.nd.array(y, ctx=ctx)
    arr_z = tvm.nd.array(z, ctx=ctx)

    matrix_mul = tvm_op.make_matrix_mul(shapeX, True, shapeY, False, tgt, tgt_host, "matrix_mul")
    matrix_mul(arr_x, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.dot(np.transpose(x), y), z, rtol=1e-5)
    
    shapeX = (500, 1000)
    shapeY = (2000, 500)
    shapeZ = (1000, 2000)   
    x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
    y = np.random.uniform(0, 10, size=shapeY).astype(dtype)
    z = np.zeros(shapeZ).astype(dtype)
    arr_x = tvm.nd.array(x, ctx=ctx)
    arr_y = tvm.nd.array(y, ctx=ctx)
    arr_z = tvm.nd.array(z, ctx=ctx)

    matrix_mul = tvm_op.make_matrix_mul(shapeX, True, shapeY, True, tgt, tgt_host, "matrix_mul")
    matrix_mul(arr_x, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.dot(np.transpose(x), np.transpose(y)), z, rtol=1e-5)


def test_conv2d():
    # im2col and np_conv2d are helper functions
    def im2col(X, filter_H, filter_W, padding, stride):
      N, C, H, W = X.shape
      assert (H + 2 * padding - filter_H) % stride == 0
      assert (W + 2 * padding - filter_W) % stride == 0
      out_H = (H + 2 * padding - filter_H) / stride + 1
      out_W = (W + 2 * padding - filter_W) / stride + 1

      y_row_size = C * filter_H * filter_W
      y_col_size = out_H * out_W
      y_shape = (N, y_row_size, y_col_size)
      Y = np.empty(y_shape, dtype = X.dtype)

      for batch_index in range(N):
        for col_index in range(y_col_size):
          out_y = col_index / out_W
          out_x = col_index % out_W
          in_y = out_y * stride - padding
          in_x = out_x * stride - padding
          row_idx = 0
          for c in range(0, C):
            for y in range(in_y, in_y + filter_H):
              for x in range(in_x, in_x + filter_W):
                if (x < 0 or x >= W or y < 0 or y >= H):
                  Y[batch_index, row_idx, col_index] = 0
                else:
                  Y[batch_index, row_idx, col_index] = X[batch_index, c, y, x]
                row_idx += 1
      return Y

    def np_conv2d(X, Filter, padding=0, stride=1):
        """Implement a conv2d as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) / stride + 1
        out_W = (W + 2 * padding - filter_W) / stride + 1

        im2col_matrix = im2col(X, filter_H, filter_W, padding, stride)
        filter_matrix = Filter.reshape(filter_outChannel, -1)
        return np.matmul(filter_matrix, im2col_matrix).reshape(N, filter_outChannel, out_H, out_W)

    shapeX = (100, 3, 28, 28)
    shapeF = (10, 3, 5, 5)
    shapeY = (100, 10, 24, 24)
    x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
    f = np.random.uniform(0, 10, size=shapeF).astype(dtype)
    y = np.zeros(shapeY).astype(dtype)
    arr_x = tvm.nd.array(x, ctx=ctx)
    arr_f = tvm.nd.array(f, ctx=ctx)
    arr_y = tvm.nd.array(y, ctx=ctx)
   
    conv2d = tvm_op.make_conv2d(shapeX, shapeF, tgt, tgt_host, "conv2d")
    conv2d(arr_x, arr_f, arr_y)
    y = arr_y.asnumpy()   
    np.testing.assert_allclose(np_conv2d(x, f), y, rtol=1e-5)


def test_relu():
    shape = (2000, 2500)
    x = np.random.uniform(-1, 1, shape).astype(dtype)
    y = np.zeros(shape).astype(dtype)
    arr_x = tvm.nd.array(x, ctx=ctx)
    arr_y = tvm.nd.array(y, ctx=ctx)
    relu = tvm_op.make_relu(shape, tgt, tgt_host, "relu")
    relu(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np.maximum(x, 0).astype(dtype), y)


def test_relu_gradient():
    shape = (2000, 2500)
    x = np.random.uniform(-1, 1, shape).astype(dtype)
    grad_x = np.random.uniform(-5, 5, shape).astype(dtype)
    y = np.zeros(shape).astype(dtype)
    arr_x = tvm.nd.array(x, ctx=ctx)
    arr_grad_x = tvm.nd.array(grad_x, ctx=ctx)
    arr_y = tvm.nd.array(y, ctx=ctx)
    relu_gradient = tvm_op.make_relu_gradient(shape, tgt, tgt_host, "relu_gradient")
    relu_gradient(arr_x, arr_grad_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(((x > 0) * grad_x).astype(dtype), y)


def test_softmax():
    shape = (400, 1000)
    x = np.random.uniform(-5, 5, shape).astype(dtype)
    y = np.zeros(shape).astype(dtype)
    arr_x = tvm.nd.array(x, ctx=ctx)
    arr_y = tvm.nd.array(y, ctx=ctx)
    matrix_softmax = tvm_op.make_matrix_softmax(shape, tgt, tgt_host, "matrix_softmax")
    matrix_softmax(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(autodiff.softmax_func(x), y, rtol=1e-5)


def test_softmax_cross_entropy():
    shape = (400, 1000)
    y = np.random.uniform(-5, 5, shape).astype(dtype)
    y_ = np.random.uniform(-5, 5, shape).astype(dtype)
    out = np.zeros((1,)).astype(dtype)
    arr_y = tvm.nd.array(y, ctx=ctx)
    arr_y_ = tvm.nd.array(y_, ctx=ctx)
    arr_out = tvm.nd.array(out, ctx=ctx)
    matrix_softmax_cross_entropy = tvm_op.make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, "softmax_cross_entropy")
    matrix_softmax_cross_entropy(arr_y, arr_y_, arr_out)
    out = arr_out.asnumpy()
    # numpy calculation
    cross_entropy = np.mean(
        -np.sum(y_ * np.log(autodiff.softmax_func(y)), axis=1), keepdims=True)
    np.testing.assert_allclose(cross_entropy, out, rtol=1e-5)


def test_reduce_sum_axis_zero():
    shape = (500, 200, 100)
    to_shape = (200, 100)
    x = np.random.uniform(-5, 5, shape).astype(dtype)
    y = np.zeros(to_shape).astype(dtype)
    arr_x = tvm.nd.array(x, ctx=ctx)
    arr_y = tvm.nd.array(y, ctx=ctx)

    reduce_sum_axis_zero = tvm_op.make_reduce_sum_axis_zero(shape, tgt, tgt_host, "reduce_sum_axis_zero")
    reduce_sum_axis_zero(arr_x, arr_y)
    
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np.sum(x, axis=0), y, rtol=1e-5)


def test_broadcast_to():
    shape = (200, 300)
    to_shape = (130, 200, 300)
    x = np.random.uniform(-1, 1, shape).astype(dtype)
    y = np.zeros(to_shape).astype(dtype)
    arr_x = tvm.nd.array(x, ctx=ctx)
    arr_y = tvm.nd.array(y, ctx=ctx)
    broadcast_to = tvm_op.make_broadcast_to(shape, to_shape, tgt, tgt_host, "broadcast_to")
    broadcast_to(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np.broadcast_to(x, to_shape), y)

