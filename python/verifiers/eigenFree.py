
import cv2

from verification.VerificationTemplate import VerificationTemplate

class EigenFree(VerificationTemplate):

        def __init__(self):
            super(
                EigenFree,
                self).__init__(
                    name='EigenFree',
                    estimates_essential=True,
                    estimates_fundamental=False)

            self.config = Config(net_depth=12, net_nchannel=128, net_act_pos="post",
                                 net_batchnorm=True,net_gcnorm=True)

            self.model = MyNetwork(self.config)

        def estimate_essential_matrix(self, kpts1, kpts2):
            """
            Estimate the Essential matrix between 2 images from a set of putative keypoint matches
            (kpt1s[i], kpts2[i]) are a corresponding matche from image 1 and 2 in normalized coordinates

            :param kpts1: Keypoints for image 1
            :type kpts1: np.array (Nx2)
            :param kpts2: Keypoints from image 2
            :type kpts2: np.array (Nx2)
            :returns: E (the 3x3 Essential matrix)
            :rtype: np.array(3x3)
            """
            def test_process(mode, sess,
                             cur_global_step, merged_summary_op, summary_writer,
                             x, y, R, t, is_training,
                             img1, img2, r,
                             logits_mean, T_hat, loss,
                             data,
                             res_dir, config, va_res_only=False):



                    _xs = np.array(
                        xs[idx_cur][:, :, :]
                    ).reshape(1, 1, -1, 8)
                    _ys = np.array(
                        ys[idx_cur][:]
                    ).reshape(1, -1, 1)
                    _dR = np.array(Rs[idx_cur]).reshape(1, 3, 9)
                    _dt = np.array(ts[idx_cur]).reshape(1, 3, 3)
                    # Create random permutation indices
                    feed_dict = {
                        x: _xs,
                        y: _ys,
                        R: _dR,
                        t: _dt,
                        is_training: config.net_bn_test_is_training,
                    }
                    fetch = {
                        "T_hat": T_hat,  # trifocal tensor by using 6 points algorithm.
                    }

                    # print("Running network for {} correspondences".format(
                    #     _xs.shape[2]
                    # ))
                    time_start = datetime.datetime.now()
                    res = sess.run(fetch, feed_dict=feed_dict)
                    time_end = datetime.datetime.now()
                    time_diff = time_end - time_start
                    # print("Runtime in milliseconds: {}".format(
                    #     float(time_diff.total_seconds() * 1000.0)
                    # ))
                    time_us += [time_diff.total_seconds() * 1000.0]
                    T_hats.append(res["T_hat"])
                    y_hats.append(res["y_hat"])

                for cur_val_idx in xrange(num_sample):
                    _xs = xs[cur_val_idx][:, :, :].reshape(1, 1, -1, 8)
                    # _ys = ys[cur_val_idx][:, :].reshape(1, -1, 1)
                    _dR = Rs[cur_val_idx].reshape(1, 3, 9)
                    _dt = ts[cur_val_idx].reshape(1, 3, 3)  # todo: these data are used for what?

                    T_hat_out = T_hats[cur_val_idx].flatten()
                    y_hat_out = y_hats[cur_val_idx].flatten()
                    if len(y_hat_out) != _xs.shape[2]:
                        y_hat_out = np.ones(_xs.shape[2])
                    # Eval decompose for all pairs
                    _xs = _xs.reshape(-1, 8)
                    # x coordinates
                    _x1 = _xs[:, :2]  # (kps, 2)
                    _x2 = _xs[:, 2:4]
                    _x3 = _xs[:, 6:]  # for the third view
                    # current validity from network
                    _valid = y_hat_out.flatten()
                    # choose top ones (get validity threshold)
                    _valid_th = np.sort(_valid)[::-1][config.obj_top_k]
                    _relu_tanh = np.maximum(0, np.tanh(_valid))

                    # For every things to test
                    _use_prob = True
                    for _test in test_list:
                        if _test == "ours":
                            _eval_func = "non-decompose"
                            _mask_before = _valid >= max(0, _valid_th)
                            _method = None
                            _probs = None
                            _weighted = False
                        elif _test == "ours_ransac":
                            _eval_func = "decompose"
                            _mask_before = _valid >= max(0, _valid_th)
                            _method = cv2.RANSAC
                            _probs = None
                            _weighted = False

                        if _eval_func == "non-decompose":
                            # this is just using DLT to get the trifocal tensor.
                            _err_q, _err_t, _, _, _num_inlier, \
                                _ = eval_nondecompose(
                                    _x1, _x2, _x3, T_hat_out, _dR, _dt, y_hat_out)
                            _mask_after = _mask_before

                        elif _eval_func == "decompose":
                            # "6 points algorithm + RANSAC loop with ours"
                            time_start = datetime.datetime.now()
                            _err_q, _err_t, _, _, _num_inlier, \
                                _mask_after = eval_decompose(
                                    _x1, _x2, _x3, _dR, _dt, mask=_mask_before,
                                    method=_method, probs=_probs,
                                    weighted=_weighted, use_prob=_use_prob)
                            time_end = datetime.datetime.now()
                            time_diff = time_end - time_start

                            # print("Runtime in milliseconds: {}".format(
                            #     float(time_diff.total_seconds() * 1000.0)
                            # ))
                            # print("RANSAC loop without ours")
                            inlier_us += [np.sum(_mask_before)]
                            inlier_ransac_us += [np.sum(_mask_after)]
                            time_ransac_us += [time_diff.total_seconds() * 1000.0]
                            time_start = datetime.datetime.now()
                            _, _, _, _, _, \
                                _mask_tmp = eval_decompose(
                                    _x1, _x2, _x3, _dR, _dt,
                                    mask=np.ones_like(_mask_before).astype(bool),
                                    method=_method, probs=_probs,
                                    weighted=_weighted, use_prob=_use_prob)
                            time_end = datetime.datetime.now()
                            time_diff = time_end - time_start
                            inlier_ransac += [np.sum(_mask_tmp)]
                            # print("Runtime in milliseconds: {}".format(
                            #     float(time_diff.total_seconds() * 1000.0)
                            # ))
                            time_ransac += [time_diff.total_seconds() * 1000.0]

                        # Load them in list
                        eval_res["err_q"][_test][cur_val_idx] = _err_q
                        eval_res["err_t"][_test][cur_val_idx] = _err_t
                        eval_res["num"][_test][cur_val_idx] = _num_inlier

                        if config.vis_dump:
                            dump_val_res(
                                img1s[cur_val_idx],
                                img2s[cur_val_idx],
                                _x1, _x2, _mask_before, _mask_after,
                                cx1s[cur_val_idx],
                                cy1s[cur_val_idx],
                                f1s[cur_val_idx],
                                cx2s[cur_val_idx],
                                cy2s[cur_val_idx],
                                f2s[cur_val_idx],
                                Rs[cur_val_idx],
                                ts[cur_val_idx],
                                os.path.join(
                                    res_dir, mode, "match", _test,
                                    "pair{:08d}".format(cur_val_idx)
                                ),
                            )




    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class MyNetwork(object):
        """Network class """

        def __init__(self, config):

            self.config = config
            self.save_file_best = dirname + '/learnedCorres_misc/models-best'
            # Initialize thenosrflow session
            self._init_tensorflow()

            # Build the network
            self._build_placeholder()
            self._build_preprocessing()
            self._build_model()
            self._build_writer()
            self._restore()

        def _init_tensorflow(self):
            # limit CPU threads with OMP_NUM_THREADS
            num_threads = os.getenv("OMP_NUM_THREADS", "")
            if num_threads != "":
                num_threads = int(num_threads)
                print("limiting tensorflow to {} threads!".format(
                    num_threads
                ))
                # Limit
                tfconfig = tf.ConfigProto(
                    intra_op_parallelism_threads=num_threads,
                    inter_op_parallelism_threads=num_threads,
                )
            else:
                tfconfig = tf.ConfigProto()

            tfconfig.gpu_options.allow_growth = True

            self.sess = tf.Session(config=tfconfig)

        def _build_placeholder(self):
            """Build placeholders."""

            # Make tensforflow placeholder
            self.x_in = tf.placeholder(tf.float32, [None, 1, None, 4], name="x_in")
            self.R_in = tf.placeholder(tf.float32, [None, 9], name="R_in")
            self.t_in = tf.placeholder(tf.float32, [None, 3], name="t_in")
            self.is_training = tf.placeholder(tf.bool, (), name="is_training")

            # Global step for optimization
            self.global_step = tf.get_variable(
                "global_step", shape=(),
                initializer=tf.zeros_initializer(),
                dtype=tf.int64,
                trainable=False)

        def _build_preprocessing(self):
            """Build preprocessing related graph."""

            # For now, do nothing
            pass

        def _build_model(self):
            """Build our MLP network."""

            with tf.variable_scope("Matchnet", reuse=tf.AUTO_REUSE):
                # For determining the runtime shape
                x_shp = tf.shape(self.x_in)

                # -------------------- Network archintecture --------------------
                # Build graph
                print("Building Graph")
                self.logits = build_graph(self.x_in, self.is_training, self.config)
                # ---------------------------------------------------------------

                # Turn into weights for each sample
                self.w = tf.nn.relu(tf.tanh(self.logits))


        def _build_writer(self):
            """Build the writers and savers"""
            # Create savers (one for current, one for best)
            self.saver_cur = tf.train.Saver()
            self.saver_best = tf.train.Saver()


        def _restore(self):
            """Test routine"""

            # Check if model exists
            if not os.path.exists(self.save_file_best + ".index"):
                print("Model File {} does not exist! Quiting".format(
                    self.save_file_best))
                exit(1)

            # Restore model
            print("Restoring from {}...".format(
                self.save_file_best))
            self.saver_best.restore(
                self.sess,
                self.save_file_best)


    def build_graph(x_in, is_training, config):

        activation_fn = tf.nn.relu

        x_in_shp = tf.shape(x_in)

        cur_input = x_in
        print(cur_input.shape)
        idx_layer = 0
        numlayer = config.net_depth
        ksize = 1
        nchannel = config.net_nchannel
        # Use resnet or simle net
        act_pos = config.net_act_pos
        conv1d_block = conv1d_resnet_block

        # First convolution
        with tf.variable_scope("hidden-input"):
            cur_input = conv1d_layer(
                inputs=cur_input,
                ksize=1,
                nchannel=nchannel,
                activation_fn=None,
                perform_bn=False,
                perform_gcn=False,
                is_training=is_training,
                act_pos="pre",
                data_format="NHWC",
            )
            print(cur_input.shape)
        for _ksize, _nchannel in zip(
                [ksize] * numlayer, [nchannel] * numlayer):
            scope_name = "hidden-" + str(idx_layer)
            with tf.variable_scope(scope_name):
                cur_input = conv1d_block(
                    inputs=cur_input,
                    ksize=_ksize,
                    nchannel=_nchannel,
                    activation_fn=activation_fn,
                    is_training=is_training,
                    perform_bn=config.net_batchnorm,
                    perform_gcn=config.net_gcnorm,
                    act_pos=act_pos,
                    data_format="NHWC",
                )
                # Apply pooling if needed
                print(cur_input.shape)

            idx_layer += 1

        with tf.variable_scope("output"):
            cur_input = conv1d_layer(
                inputs=cur_input,
                ksize=1,
                nchannel=1,
                activation_fn=None,
                is_training=is_training,
                perform_bn=False,
                perform_gcn=False,
                data_format="NHWC",
            )
            #  Flatten
            cur_input = tf.reshape(cur_input, (x_in_shp[0], x_in_shp[2]))

        logits = cur_input
        print(cur_input.shape)

        return logits

    from six.moves import xrange


    # ------------------------------------------------------------
    # Tensorflow ops

    def tf_get_shape_as_list(x):

        return [_s if _s is not None else - 1 for _s in x.get_shape().as_list()]


    def tf_quaternion_from_matrix(M):



        m00 = M[:, 0, 0][..., None]
        m01 = M[:, 0, 1][..., None]
        m02 = M[:, 0, 2][..., None]
        m10 = M[:, 1, 0][..., None]
        m11 = M[:, 1, 1][..., None]
        m12 = M[:, 1, 2][..., None]
        m20 = M[:, 2, 0][..., None]
        m21 = M[:, 2, 1][..., None]
        m22 = M[:, 2, 2][..., None]
        # symmetric matrix K
        zeros = tf.zeros_like(m00)
        K = tf.concat(
            [m00 - m11 - m22, zeros, zeros, zeros,
             m01 + m10, m11 - m00 - m22, zeros, zeros,
             m02 + m20, m12 + m21, m22 - m00 - m11, zeros,
             m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            axis=1)
        K = tf.reshape(K, (-1, 4, 4))
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = tf.self_adjoint_eig(K)

        q0 = V[:, 3, 3][..., None]
        q1 = V[:, 0, 3][..., None]
        q2 = V[:, 1, 3][..., None]
        q3 = V[:, 2, 3][..., None]
        q = tf.concat([q0, q1, q2, q3], axis=1)
        sel = tf.reshape(tf.to_float(q[:, 0] < 0.0), (-1, 1))
        q = (1.0 - sel) * q - sel * q

        return q


    def tf_matrix_from_quaternion(q, eps=1e-10):



        # Make unit quaternion
        q_norm = q / (eps + tf.norm(q, axis=1, keep_dims=True))
        q_norm *= tf.constant(2.0 ** 0.5, dtype=tf.float32)
        qq = tf.matmul(
            tf.reshape(q_norm, (-1, 4, 1)),
            tf.reshape(q_norm, (-1, 1, 4))
        )
        M = tf.stack([
            1.0 - qq[:, 2, 2] - qq[:, 3, 3], qq[:, 1, 2] - qq[:, 3, 0],
            qq[:, 1, 3] + qq[:, 2, 0], qq[:, 1, 2] + qq[:, 3, 0],
            1.0 - qq[:, 1, 1] - qq[:, 3, 3], qq[:, 2, 3] - qq[:, 1, 0],
            qq[:, 1, 3] - qq[:, 2, 0], qq[:, 2, 3] + qq[:, 1, 0],
            1.0 - qq[:, 1, 1] - qq[:, 2, 2]
        ], axis=1)

        return M


    def tf_skew_symmetric(v):



        zero = tf.zeros_like(v[:, 0])

        M = tf.stack([
            zero, -v[:, 2], v[:, 1],
            v[:, 2], zero, -v[:, 0],
            -v[:, 1], v[:, 0], zero,
        ], axis=1)

        return M


    def tf_unskew_symmetric(M):



        v = tf.stack([
            0.5 * (M[:, 7] - M[:, 5]),
            0.5 * (M[:, 2] - M[:, 6]),
            0.5 * (M[:, 3] - M[:, 1]),
        ], axis=1)

        return v


    # ------------------------------------------------------------
    # Architecture related

    def bn_act(linout, perform_gcn, perform_bn, activation_fn, is_training,
               data_format):



        """ Perform batch normalization and activation """
        if data_format == "NHWC":
            axis = -1
        else:
            axis = 1

        # Global Context normalization on the input
        if perform_gcn:
            # Epsilon to be used in the tf.nn.batch_normalization
            var_eps = 1e-3
            # get mean variance for single sample (channel-wise, note that we omit
            # axis=1 since we are expecting a size of 1 in that dimension)
            mean, variance = tf.nn.moments(linout, axes=[2], keep_dims=True)
            # Use tensorflow's nn.batchnorm
            linout = tf.nn.batch_normalization(
                linout, mean, variance, None, None, var_eps)

        if perform_bn:
            with tf.variable_scope("bn"):
                linout = tf.layers.batch_normalization(
                    inputs=linout,
                    center=False, scale=False,
                    training=is_training,
                    trainable=True,
                    axis=axis,
                )

        if activation_fn is None:
            output = linout
        else:
            output = activation_fn(linout)

        return output


    def pad_cyclic(tensor, paddings):



        ndim = len(paddings)
        for _dim, _pad in zip(xrange(ndim), paddings):

            pad_list = []
            if _pad[0] > 0:
                # Padding to put at front
                slice_st = [slice(None, None)] * ndim
                slice_st[_dim] = slice(-_pad[0], None)
                pad_list += [tensor[tuple(slice_st)]]

            # Original
            pad_list += [tensor]

            if _pad[1] > 0:
                # Padding to put at back
                slice_ed = [slice(None, None)] * ndim
                slice_ed[_dim] = slice(None, _pad[1])
                pad_list += [tensor[tuple(slice_ed)]]

            if len(pad_list) > 1:
                # Concatenate to do padding
                tensor = tf.concat(pad_list, axis=_dim)

        return tensor


    def conv1d_pad_cyclic(inputs, ksize, numconv, data_format="NCHW"):
        in_shp = tf_get_shape_as_list(inputs)
        ksize = 2 * (ksize // 2 * numconv) + 1

        if data_format == "NCHW":
            assert (ksize < in_shp[-1]) or (in_shp[-1] == -1)
            if np.mod(ksize, 2) == 0:
                paddings = [
                    [0, 0], [0, 0], [0, 0], [ksize // 2 - 1, ksize // 2]
                ]
            else:
                paddings = [
                    [0, 0], [0, 0], [0, 0], [ksize // 2, ksize // 2]
                ]
        else:
            assert (ksize < in_shp[-2]) or (in_shp[-2] == -1)
            if np.mod(ksize, 2) == 0:
                paddings = [
                    [0, 0], [0, 0], [ksize // 2 - 1, ksize // 2], [0, 0]
                ]
            else:
                paddings = [
                    [0, 0], [0, 0], [ksize // 2, ksize // 2], [0, 0]
                ]
        inputs = pad_cyclic(inputs, paddings)

        return inputs


    def get_W_b_conv1d(in_channel, out_channel, ksize, dtype=None):



        if dtype is None:
            dtype = tf.float32

        fanin = in_channel * ksize
        W = tf.get_variable(
            "weights", shape=[1, ksize, in_channel, out_channel], dtype=dtype,
            initializer=tf.truncated_normal_initializer(stddev=2.0 / fanin),
            # initializer=tf.random_normal_initializer(stddev=0.02),
        )
        b = tf.get_variable(
            "biases", shape=[out_channel], dtype=dtype,
            initializer=tf.zeros_initializer(),
        )
        # tf.summary.histogram("W", W)
        # tf.summary.histogram("b", b)

        return W, b


    def conv1d_layer(inputs, ksize, nchannel, activation_fn, perform_bn,
                     perform_gcn, is_training, perform_kron=False,
                     padding="CYCLIC", data_format="NCHW",
                     act_pos="post"):



        assert act_pos == "pre" or act_pos == "post"

        # Pad manually
        if padding == "CYCLIC":
            if ksize > 1:
                inputs = conv1d_pad_cyclic(
                    inputs, ksize, 1, data_format=data_format)
            cur_padding = "VALID"
        else:
            cur_padding = padding

        in_shp = tf_get_shape_as_list(inputs)
        if data_format == "NHWC":
            in_channel = in_shp[-1]
            ksizes = [1, 1, ksize, 1]
        else:
            in_channel = in_shp[1]
            ksizes = [1, 1, 1, ksize]

        assert len(in_shp) == 4

        # # Lift with kronecker
        # if not is_first:
        #     inputs = tf.concat([
        #         inputs,
        #         kronecker_layer(inputs),
        #     ], axis=-1)

        pool_func = None
        self_ksize = ksize
        do_add = False

        # If pre activation
        if act_pos == "pre":
            inputs = bn_act(inputs, perform_gcn, perform_bn, activation_fn,
                            is_training, data_format)

        # Normal convolution
        with tf.variable_scope("self-conv"):
            W, b = get_W_b_conv1d(in_channel, nchannel, self_ksize)
            # Convolution in the valid region only
            linout = tf.nn.conv2d(
                inputs, W, [1, 1, 1, 1], cur_padding, data_format=data_format)
            linout = tf.nn.bias_add(linout, b, data_format=data_format)
        # Pooling Convolution for the summary route
        if pool_func is not None:
            with tf.variable_scope("neigh-conv"):
                if not do_add:
                    linout = pool_func(
                        linout,
                        ksize=ksizes,
                        strides=[1, 1, 1, 1],
                        padding=cur_padding,
                        data_format=data_format,
                    )
                else:
                    W_n, b_n = get_W_b_conv1d(in_channel, nchannel, 1)
                    # Convolution in the valid region only
                    linout_n = tf.nn.conv2d(
                        inputs, W_n, [1, 1, 1, 1], "VALID", data_format=data_format
                    )
                    linout_n = tf.nn.bias_add(
                        linout_n, b_n, data_format=data_format)
                    linout_n = pool_func(
                        linout_n,
                        ksize=ksizes,
                        strides=[1, 1, 1, 1],
                        padding=cur_padding,
                        data_format=data_format,
                    )
                    # # Crop original linout
                    # if ksize > 1:
                    #     if np.mod(ksize, 2) == 0:
                    #         crop_st = ksize // 2 - 1
                    #     else:
                    #         crop_st = ksize // 2
                    #         crop_ed = ksize // 2
                    #     linout = linout[:, :, :, crop_st:-crop_ed]
                    # Add to the original output
                    linout = linout + linout_n

        # If post activation
        output = linout
        if act_pos == "post":
            output = bn_act(linout, perform_gcn, perform_bn, activation_fn,
                            is_training, data_format)

        return output


    def conv1d_resnet_block(inputs, ksize, nchannel, activation_fn, is_training,
                            midchannel=None, perform_bn=False, perform_gcn=False,
                            padding="CYCLIC", act_pos="post", data_format="NCHW"):



        # In case we want to do a bottleneck layer
        if midchannel is None:
            midchannel = nchannel

        # don't activate conv1 in case of midact
        conv1_act_fn = activation_fn
        if act_pos == "mid":
            conv1_act_fn = None
            act_pos = "pre"

        # Pass branch
        with tf.variable_scope("pass-branch"):
            # passthrough to be used when num_outputs != num_inputs
            in_shp = tf_get_shape_as_list(inputs)
            if data_format == "NHWC":
                in_channel = in_shp[-1]
            else:
                in_channel = in_shp[1]
            if in_channel != nchannel:
                cur_in = inputs
                # Simply change channels through 1x1 conv
                with tf.variable_scope("conv"):
                    cur_in = conv1d_layer(
                        inputs=inputs, ksize=1,
                        nchannel=nchannel,
                        activation_fn=None,
                        perform_bn=False,
                        perform_gcn=False,
                        is_training=is_training,
                        padding=padding,
                        data_format=data_format,
                    )
                orig_inputs = cur_in
            else:
                orig_inputs = inputs

        # Conv branch
        with tf.variable_scope("conv-branch"):
            cur_in = inputs
            # Do bottle neck if necessary (Linear)
            if midchannel != nchannel:
                with tf.variable_scope("preconv"):
                    cur_in = conv1d_layer(
                        inputs=cur_in, ksize=1,
                        nchannel=nchannel,
                        activation_fn=None,
                        perform_bn=False,
                        perform_gcn=False,
                        is_training=is_training,
                        padding=padding,
                        data_format=data_format,
                    )
                    cur_in = activation_fn(cur_in)

            # Main convolution
            with tf.variable_scope("conv1"):
                # right branch
                cur_in = conv1d_layer(
                    inputs=cur_in, ksize=ksize,
                    nchannel=nchannel,
                    activation_fn=conv1_act_fn,
                    perform_bn=perform_bn,
                    perform_gcn=perform_gcn,
                    is_training=is_training,
                    padding=padding,
                    act_pos=act_pos,
                    data_format=data_format,
                )

            # Main convolution
            with tf.variable_scope("conv2"):
                # right branch
                cur_in = conv1d_layer(
                    inputs=cur_in, ksize=ksize,
                    nchannel=nchannel,
                    activation_fn=activation_fn,
                    perform_bn=perform_bn,
                    perform_gcn=perform_gcn,
                    is_training=is_training,
                    padding=padding,
                    act_pos=act_pos,
                    data_format=data_format,
                )

            # Do bottle neck if necessary (Linear)
            if midchannel != nchannel:
                with tf.variable_scope("postconv"):
                    cur_in = conv1d_layer(
                        inputs=cur_in, ksize=1,
                        nchannel=nchannel,
                        activation_fn=None,
                        perform_bn=False,
                        perform_gcn=False,
                        is_training=is_training,
                        padding=padding,
                        data_format=data_format,
                    )
                    cur_in = activation_fn(cur_in)

        # Crop lb or rb accordingly
        if padding == "VALID" and ksize > 1:
            # Crop pass branch results
            if np.mod(ksize, 2) == 0:
                crop_st = ksize // 2 - 1
            else:
                crop_st = ksize // 2
                crop_ed = ksize // 2
                if data_format == "NHWC":
                    orig_inputs = orig_inputs[:, :,  crop_st:-crop_ed, :]
                else:
                    orig_inputs = orig_inputs[:, :, :, crop_st:-crop_ed]

        return cur_in + orig_inputs
