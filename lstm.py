import json
import keras_nlp
import tensorflow_datasets as tfds
import tensorflow as tf


data = []
text = []
label = []
train_size = 70

with open('Sarcasm_Headlines_Dataset_v2.json') as file:
    for line in file:
        data.append(json.loads(line))


for line in data:
    text.append(line['headline'])
    label.append(line['is_sarcastic'])

train_data = text[:int((70/100)*len(text))]
test_data = text[int((70/100)*len(text)):]

train_label = label[:int((70/100)*len(text))]
test_label = label[int((70/100)*len(text)):]


train_dataset = tf.data.Dataset.from_tensor_slices(train_data)

vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
    train_dataset,
    vocabulary_size = 10000,
    reserved_tokens = ["[PAD]","[UNK]"]
)


subword_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary = vocab
)

train_sequences = train_dataset.map(lambda x : subword_tokenizer(x))
padded_train = tf.keras.utils.pad_sequences(train_sequences,padding='post',maxlen=120)

train_label = tf.convert_to_tensor(train_label)


test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
test_sequences = test_dataset.map(lambda x : subword_tokenizer(x))
paddes_test = tf.keras.utils.pad_sequences(test_sequences,padding='post',maxlen=120)

test_label = tf.convert_to_tensor(test_label)

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(120,)),
        tf.keras.layers.Embedding(10000,16),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ]
)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x=padded_train,y=train_label,validation_data=(paddes_test,test_label),epochs=10)