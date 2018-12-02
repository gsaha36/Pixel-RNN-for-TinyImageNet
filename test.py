if mask_type == 'a' and self.channel == 3:
    R, G, B = tf.split(3, 3, inputs, name='split')
    assert num_output % 3 == 0
    num_channel_out = int(num_outputs/3)
    mask = np.ones((kernel_h, kernel_w, channel, num_channel_out), dtype=np.float32)
    mask[center_h, center_w+1:, 0, :] = 0.
    mask[center_h+1:,        :, 0, :] = 0.
    mask[center_h,    center_w, 0, :] = 0.
    mask[:, : ,1, :] = 0.
    mask[:, :, 1, :] = 0.
    mask[:, :, 1, :] = 0.
    mask[:, : ,2, :] = 0.
    mask[:, :, 2, :] = 0.
    mask[:, :, 2, :] = 0.

    mask[center_h   , center_w+1:, 1, :] = 0.
    mask[center_h+1:,           :, 1, :] = 0.
    mask[center_h   , center_w   , 1, :] = 0.

    mask[:, :, 2, :] = 0.
    mask[:, :, 2, :] = 0.
    mask[:, :, 2, :] = 0.

    mask[center_h, center_w+1: ,2 ,:] = 0.
    mask[center_h+1:, :, 2, :] = 0.
    mask[center_h,center_w,2,:] = 0.


    weights_shapeR = [kernel_h, kernel_w, 1, num_channel_out]
    weightsR = tf.get_variable("weightsR", weights_shapeR,
      tf.float32, weights_initializer, weights_regularizer)
    weightsR *= tf.constant(maskR, dtype=tf.float32)
    tf.add_to_collection('conv2d_weights_%s' % mask_type, weightsR)

    outputsR = tf.nn.conv2d(R, weightsR, [1, stride_h, stride_w, 1], padding=padding, name='outputs')
    tf.add_to_collection('conv2d_outputs', outputsR)

    maskR[center_h+1:, :, :, :] = 0.
    maskR[center_h,center_w,:,:] = 0.

      weights *= tf.constant(mask, dtype=tf.float32)
      tf.add_to_collection('conv2d_weights_%s' % mask_type, weights)

    outputs = tf.nn.conv2d(inputs,
        weights, [1, stride_h, stride_w, 1], padding=padding, name='outputs')
    tf.add_to_collection('conv2d_outputs', outputs)


