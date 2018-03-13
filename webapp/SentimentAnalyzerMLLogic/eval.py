#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn
import csv
import json



def createJSONOutput(array):
    sentiments = {}
    for arr in array:
      sentiments[str(arr[0])] = {
          'sentiment' : str(arr[1])
      }
    print(sentiments)
    return sentiments
# Parameters
# ==================================================

# Data Parameters
def doEval(array1):

    array = []
    if type(array1) is str:
        array.append(array1)
    else:
        array = array1




    checkpoint_dir = "/Users/ankeshkatiyar/Tensorflow/bin/SentimentAnalyzer1/webapp/SentimentAnalyzerMLLogic/runs/1520925469/checkpoints"


    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # CHANGE THIS: Load data. Load your own data here
    # if FLAGS.eval_train:
    #     x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    #     y_test = np.argmax(y_test, axis=1)
    # else:
    #     x_raw = ["SAP products are great", "Sap products are not good."]
    #     #y_test = [1, 0]
    x_raw = array
    # Map data into vocabulary
    #vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_path = "/Users/ankeshkatiyar/Tensorflow/bin/SentimentAnalyzer1/webapp/SentimentAnalyzerMLLogic/runs/1520925469/vocab"
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    print("\nEvaluating...\n")

# Evaluation
# ==================================================

    y_test = []
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement= True,
          log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), 64, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])



    # Print accuracy if y_test is defined
    # if y_test is not None:
    #     correct_predictions = float(sum(all_predictions == y_test))
    #     print("Total number of test examples: {}".format(len(y_test)))
    #     print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))

    return createJSONOutput(predictions_human_readable)

doEval("Ankesh")
