
import tensorflow as tf
import numpy as np
# corpus = []

with open('Laurences_generated_poetry.txt','r') as file:
    f = file.read()
    corpus = f.lower().split('\n')
print(corpus)

vectorize_layer = tf.keras.layers.TextVectorization()

vectorize_layer.adapt(corpus)

vocabulary = vectorize_layer.get_vocabulary()

vocab_size = len(vocabulary)


input_sequence = []

for line in corpus:
    sequence = vectorize_layer(line).numpy()
    for i in range(1,len(sequence)):
        n_gram_sequence = sequence[:i+1]
        input_sequence.append(n_gram_sequence)

    '''
    [1,2]
    [1,2,3]
    [1,2,3,4]
    [1,2,3,4,5]
    '''

max_sequence = max([len(x) for x in input_sequence])


pad_seq = tf.keras.utils.pad_sequences(input_sequence,padding='pre',maxlen=max_sequence)

x = pad_seq[:,:-1]
label = pad_seq[:,-1]

ys = tf.keras.utils.to_categorical(label,num_classes=vocab_size)


model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(max_sequence-1,)),
        tf.keras.layers.Embedding(vocab_size,100),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(vocab_size,activation='softmax')
    ]
)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x=x,y=ys,epochs=100)

example = "some text"

word_length = 100

for _ in range(word_length):

    seq = vectorize_layer(example)

    example_pad = tf.keras.utils.pad_sequences(
        [seq],
        maxlen=max_sequence,
        padding='pre'
    )

    prob = model.predict(example_pad,verbose=0)

    pred = np.argmax(prob,axis=-1)[0]

    output_word = vocabulary[pred]

    example += ' '+output_word

print(example)