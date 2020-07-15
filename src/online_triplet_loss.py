import tensorflow as tf

class Online_Mining(tf.keras.layers.Layer):

    def __init__(self,margin = 0.6):
        super(Online_Mining, self).__init__()
        self.margin = margin

    def call(self, total_input):

        self.inputs,y_inputs = total_input
        self.y_inputs = tf.reshape(y_inputs,[y_inputs.shape[0],])

        dot_product = tf.matmul(self.inputs, tf.transpose(self.inputs))
        square_norm = tf.linalg.diag_part(dot_product)
        distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
        pairwise_dist = tf.maximum(distances,0.0)

        mask_anchor_positive = tf.cast(positive_anchor_mask(self.y_inputs),tf.float32)
        anchor_positive_dist = tf.multiply(pairwise_dist,mask_anchor_positive)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

        mask_anchor_negative = tf.cast(negative_anchor_mask(self.y_inputs),tf.float32)
        anchor_negative_dist_max = tf.reduce_max(pairwise_dist,axis = 1,keepdims=True)
        anchor_negative_dist = pairwise_dist + tf.multiply(anchor_negative_dist_max,(1.0 - mask_anchor_negative))
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist,axis = 1, keepdims = True )

        loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + self.margin, 0.0)
        loss = tf.reduce_mean(loss)

        return loss

def positive_anchor_mask(y_inputs):
  '''
  Returns a mask of positive anchor pairs i.e, True if anchor != positive and labels(anchor) == labels(positive) else False
  '''
  equals_positive = tf.equal(tf.expand_dims(y_inputs,0),tf.expand_dims(y_inputs,1))
  indices_not_equals = tf.logical_not(tf.cast(tf.eye(tf.shape(y_inputs)[0]),tf.bool))
  anchor_positive_mask = tf.logical_and(equals_positive,indices_not_equals)

  return anchor_positive_mask

def negative_anchor_mask(y_inputs):
  '''
  Returns a mask of valid negative anchor pairs i.e, True if anchor != negative and labels(anchor) != labels(negative) else False
  '''
  negative_anchor_mask = tf.logical_not(tf.equal(tf.expand_dims(y_inputs,0),tf.expand_dims(y_inputs,1)))
  return negative_anchor_mask

