import tensorflow as tf


'''
    by default padding process is done by Text vectorization module
    and it do post padding. so inorder to do pre padding, we have to 
    follow below method
'''

sentences = [
    'Hello how are you',
    'Good morning',
    'Natural Language processing using Tensorflow....'
]



#initialize text vectorize
vectorize_layer = tf.keras.layers.TextVectorization()

#create a vocabulary
vectorize_layer.adapt(sentences)

#get vocabulary
vocabulary = vectorize_layer.get_vocabulary(include_special_tokens=False)


#create tensorflow dataset
sentence_dataset = tf.data.Dataset.from_tensor_slices(sentences)


#map textvector to tensorflow dataset
sequences = sentence_dataset.map(vectorize_layer)

#pad sequence using pad_sequence module and do pre padding
sequence_pre = tf.keras.utils.pad_sequences(sequences,padding='pre')

print(sequence_pre)

#method 2

'''
    ragged tensor means tensor that are different shapes.we are telling 
    layer that keep the intial length of sequences same length and not pad
    automatically
'''
vectorize_layer_ = tf.keras.layers.TextVectorization(ragged=True)

vectorize_layer_.adapt(sentences)

ragged_sentences = vectorize_layer_(sentences)

print(ragged_sentences)

pre_pad_sequence = tf.keras.utils.pad_sequences(ragged_sentences.numpy())

print(pre_pad_sequence)


'''
    other parameters in pad_sequences 
    maxlen - maximum length of each sentences
    truncating - slice sentence (pre,post)
'''
