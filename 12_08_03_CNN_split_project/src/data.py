# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:24:03 2018

@author: Sandalfon
"""
import os
import tensorflow as tf

class Data(object):
    
    def __load(self,options,filename_queue):
        image_height = options.getDataOptions('img_height')
        image_width = options.getDataOptions('img_width')
        num_channels = options.getDataOptions('num_channels')
        distort_images = options.getDataOptions('distort_images')
        crop_width = options.getDataOptions('crop_width')
        crop_height = options.getDataOptions('crop_height')
        image_vec_length = image_height * image_width * num_channels
        record_length = 1 + image_vec_length # ( + 1 for the 0-9 label)
        reader = tf.FixedLengthRecordReader(record_bytes=record_length)
        key, record_string = reader.read(filename_queue)
        record_bytes = tf.decode_raw(record_string, tf.uint8)
        image_label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)
        # Extract image
        image_extracted = tf.reshape(
                tf.slice(record_bytes, [1], [image_vec_length]),
                                 [num_channels, image_height, image_width])
        # Reshape image
        image_uint8image = tf.transpose(image_extracted, [1, 2, 0])
        reshaped_image = tf.cast(image_uint8image, tf.float32)
        # Randomly Crop image
        final_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, crop_width, crop_height)
        if distort_images:
            # Randomly flip the image horizontally, change the brightness and contrast
            final_image = tf.image.random_flip_left_right(final_image)
            final_image = tf.image.random_brightness(final_image,max_delta=63)
            final_image = tf.image.random_contrast(final_image,lower=0.2, upper=1.8)
        
        # Normalize whitening
        final_image = tf.image.per_image_standardization(final_image)
        return(final_image, image_label)
    
    def batches(self, options, train_logical=True):
        data_dir = options.getDataOptions('data_dir')
        extract_folder = options.getDataOptions('extract_folder')
        batch_size = options.getRunOptions('batch_size')
        min_after_dequeue  = options.getRunOptions('min_after_dequeue')
        if train_logical:
            files = [os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(i)) for i in range(1,6)]
        else:
            files = [os.path.join(data_dir, extract_folder, 'test_batch.bin')]
        filename_queue = tf.train.string_input_producer(files)
        image, label = self.__load(options,filename_queue)
        capacity = min_after_dequeue + 3 * batch_size
        example_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
        return (example_batch, label_batch)