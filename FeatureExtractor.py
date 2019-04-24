from config.config import FLAGS
import tensorflow as tf
import numpy as np
import importlib
import cv2
import time


class FeatureExtractor:

    def __init__(self):
        self._build_model()
        # Create kalman filters
        if FLAGS.use_kalman:
            self.kalman_filter_array = [cv2.KalmanFilter(4, 2) for _ in range(FLAGS.num_of_joints)]
            for _, joint_kalman_filter in enumerate(self.kalman_filter_array):
                joint_kalman_filter.transitionMatrix = np.array(
                    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                    np.float32)
                joint_kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                joint_kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                               np.float32) * FLAGS.kalman_noise

    def predict(self, img):
        # inference
        t1 = time.time()
        stage_heatmap_np = self.sess.run([self.output_node],
                                    feed_dict={self.model.input_images: img})
        print('FPS: %.2f' % (1 / (time.time() - t1)))

        # interpret data
        last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.num_of_joints].reshape(
            (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
        last_heatmap = cv2.resize(last_heatmap, (FLAGS.input_size, FLAGS.input_size))

        if self.kalman_filter_array is not None:
            for joint_num in range(FLAGS.num_of_joints):
                tmp_heatmap = stage_heatmap_np[:, :, joint_num]
                joint_coord = np.unravel_index(np.argmax(tmp_heatmap),
                                               (FLAGS.input_size, FLAGS.input_size))
                mean_response_val += tmp_heatmap[joint_coord[0], joint_coord[1]]
                joint_coord = np.array(joint_coord).reshape((2, 1)).astype(np.float32)
                kalman_filter_array[joint_num].correct(joint_coord)
                kalman_pred = kalman_filter_array[joint_num].predict()
                correct_coord = np.array([kalman_pred[0], kalman_pred[1]]).reshape((2))
                local_joint_coord_set[joint_num, :] = correct_coord

                # Resize back
                correct_coord /= crop_full_scale

                # Substract padding border
                correct_coord[0] -= (tracker.pad_boundary[0] / crop_full_scale)
                correct_coord[1] -= (tracker.pad_boundary[2] / crop_full_scale)
                correct_coord[0] += tracker.bbox[0]
                correct_coord[1] += tracker.bbox[2]
                joint_coord_set[joint_num, :] = correct_coord

        else:
            for joint_num in range(FLAGS.num_of_joints):
                tmp_heatmap = stage_heatmap_np[:, :, joint_num]
                joint_coord = np.unravel_index(np.argmax(tmp_heatmap),
                                               (FLAGS.input_size, FLAGS.input_size))
                mean_response_val += tmp_heatmap[joint_coord[0], joint_coord[1]]
                joint_coord = np.array(joint_coord).astype(np.float32)

                local_joint_coord_set[joint_num, :] = joint_coord

                # Resize back
                joint_coord /= crop_full_scale

                # Substract padding border
                joint_coord[0] -= (tracker.pad_boundary[2] / crop_full_scale)
                joint_coord[1] -= (tracker.pad_boundary[0] / crop_full_scale)
                joint_coord[0] += tracker.bbox[0]
                joint_coord[1] += tracker.bbox[2]
                joint_coord_set[joint_num, :] = joint_coord

    # build the model
    def _build_model(self):
        cpm_model = importlib.import_module('models.net.' + FLAGS.network_def)
        self.model = cpm_model.CPM_Model(input_size=FLAGS.input_size,
                                         heatmap_size=FLAGS.heatmap_size,
                                         stages=FLAGS.cpm_stages,
                                         joints=FLAGS.num_of_joints,
                                         img_type=FLAGS.color_channel,
                                         is_training=False)
        self.output_node = tf.get_default_graph().get_tensor_by_name(name=FLAGS.output_node_names)

    def _init_model(self):
        saver = tf.train.Saver()
        device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
        sess_config = tf.ConfigProto(device_count=device_count)
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True
        self.sess = tf.Session(config=sess_config)
        if FLAGS.model_path.endswith('pkl'):
            self.model.load_weights_from_file(FLAGS.model_path, self.sess, False)
        else:
            saver.restore(self.sess, FLAGS.model_path)

    def __enter__(self):
        self._init_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()



