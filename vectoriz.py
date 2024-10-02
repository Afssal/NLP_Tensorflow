import tensorflow as tf

sentences = [
    'Hello how are you',
    'Good morning',
    'Natural Language processing using Tensorflow....'
]

#initialize TextVectorization
vectorize_layer = tf.keras.layers.TextVectorization()

#creating vocabulary
vectorize_layer.adapt(sentences)

#get vocabulary 
vocabulary = vectorize_layer.get_vocabulary(include_special_tokens=False)

print(vocabulary)


'''
    if we didnt set include_special_tokens to false, by default its value is true and
    we get value two special tokens in vocabulary that is [UNK] and ''

'''

vocabulary_ = vectorize_layer.get_vocabulary()

print(vocabulary_)

#iterate each token in vocabulary
for index,token in enumerate(vocabulary):
    print(index,token)

#convert a text to sequence
sequence = vectorize_layer('Hello Good morining')


print(sequence)

#convert multiple text to sequence
sequence_ = vectorize_layer(sentences)

print(sequence_)

'''
    from the above output we can see that padding is added automatically
'''