# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import random
import numpy as np
from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

RANDOM_SEED = 1024
LABEL_MAP = {"person":1,
             "people":1,
             "cyclist":1,
             "person?":1
             }
IMAGE_SHAPE = [512,640,3] ##Height, Width, Depth



flags = tf.app.flags
flags.DEFINE_string('data_dir', '../../../data-kaist/', 'Root directory to raw KAIST dataset.')
flags.DEFINE_string('imageset_dir', '../../../data-kaist/imageSets/train01_valid.txt', 'directory to raw KAIST image set.')
flags.DEFINE_string('annotations_dir', '../../../data-kaist/annotations/', 'directory to raw KAIST annotations.')
flags.DEFINE_string('output_path', '../KAIST/TF_DATA', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', '../data/KAIST_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_string('image_type', 'visible',
                    'image type')
flags.DEFINE_integer('skip', 2,
                    'sample rate for video')
flags.DEFINE_boolean('shuffle', True,
                    'random shuffle the order ')
FLAGS = flags.FLAGS

def _annotation_paser(annotation_dir):
    labels_text = []
    labels =[]
    bboxes = []
    with open(annotation_dir) as f:
        for line in f:
            line = line.strip().split()
            if line[0] == "%":
                continue
            else:
                labels_text.append("person".encode('utf8'))
                labels.append(int(LABEL_MAP[line[0]]))
                box = [int(i) for i in line[1:5]]
                ##bbox format "xywh"
                ## convert to "xmin,ymin, xmax, ymax"
                x,y,w,h = box
                xmin = float(x)/float(IMAGE_SHAPE[1])
                ymin = float(y)/float(IMAGE_SHAPE[0])
                xmax = np.minimum(1.0,float(x + w)/float(IMAGE_SHAPE[1]))
                ymax = np.minimum(1.0,float(y + h)/float(IMAGE_SHAPE[0]))
                box = [ymin,xmin,ymax,xmax]
                bboxes.append(box)
    return labels_text ,labels,bboxes

def _write_to_tf_record(image_dir,annotation_dir,tf_record_writer):

    with tf.gfile.GFile(image_dir, 'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = PIL.Image.open(encoded_png_io)
    image = np.asarray(image)

    key = hashlib.sha256(encoded_png).hexdigest()


    labels_text, labels, bboxes = _annotation_paser(annotation_dir)

    xmin = []
    ymin = []
    xmax = []
    ymax = []

    # split the bboxes into individual
    for box in bboxes:
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], box)]

    example = tf.train.Example(features=tf.train.Features(feature = {
        'image/height': dataset_util.int64_feature(IMAGE_SHAPE[0]),
        'image/width': dataset_util.int64_feature(IMAGE_SHAPE[1]),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
        'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
        'image/object/bbox/xmin':dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/label':dataset_util.int64_list_feature(labels),
        'image/object/class/text':dataset_util.bytes_list_feature(labels_text)}))

    tf_record_writer.write(example.SerializeToString())

def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue

    difficult_obj.append(int(difficult))

    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main(_):
    imageset_dir = FLAGS.imageset_dir
    if not tf.gfile.Exists(imageset_dir):
        raise Exception("no such imagelist path")
    if not tf.gfile.Exists(FLAGS.output_path):
        tf.gfile.MkDir(FLAGS.output_path)
    with open(imageset_dir) as f:
        imagelist = [i.strip() for i in f]

    ### sample data
    imagelist = [v for i, v in enumerate(imagelist) if i % FLAGS.skip == 0]

    ### shulffe data
    if FLAGS.shuffle:
        random.seed(RANDOM_SEED)
        random.shuffle(imagelist)
    imagelist_dir, _ = os.path.splitext(imageset_dir)
    file_name = imagelist_dir.split('/')[-1]
    tf_file_name = "{}/{}_skip{}.tf".format(FLAGS.output_path, file_name, FLAGS.skip)




    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)


    with tf.python_io.TFRecordWriter(tf_file_name) as tf_record_writer:
        for idx, example in enumerate(imagelist):
          if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(imagelist))
          name = example.split('/')
          image_dir = os.path.join(FLAGS.data_dir, name[0], name[1], FLAGS.image_type, name[2] + ".jpg")
          annotation_dir = os.path.join(FLAGS.annotations_dir, name[0], name[1], name[2] + ".txt")
          _write_to_tf_record(image_dir, annotation_dir, tf_record_writer)


if __name__ == '__main__':
  tf.app.run()
