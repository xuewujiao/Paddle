cc_test(
  test_mkldnn_op_nhwc
  SRCS mkldnn/test_mkldnn_op_nhwc.cc
  DEPS op_registry
       pool_op
       shape_op
       crop_op
       activation_op
       pooling
       transpose_op
       scope
       device_context
       enforce
       executor)
