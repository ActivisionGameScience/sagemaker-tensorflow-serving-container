import random
import tensorflow as tf



with tf.Session() as sess:
    inputs = tf.placeholder(tf.string, [None])
    x = tf.map_fn(lambda x: tf.image.decode_png(x, channels=3), inputs, dtype=tf.uint8)
    x = tf.cast(x, tf.float32)
    x = tf.reshape(x, [-1, 32*32*3])
    x = tf.matmul(x, tf.random_normal([32*32*3, 10]))
    outputs = tf.nn.softmax(x)

    inputs_info = tf.saved_model.utils.build_tensor_info(inputs)
    inputs_def = {tf.saved_model.signature_constants.PREDICT_INPUTS: inputs_info}
    outputs_info = tf.saved_model.utils.build_tensor_info(outputs)
    outputs_def = {tf.saved_model.signature_constants.PREDICT_OUTPUTS: outputs_info}

    signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs_def,
            outputs=outputs_def,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
    )

    builder = tf.saved_model.builder.SavedModelBuilder('cifar_image/0')
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
    )
    builder.save()
