import keras_nlp
import tensorflow_datasets as tfds
import tensorflow as tf

imdb,info = tfds.load('imdb_reviews',with_info=True,as_supervised=True,data_dir='./data',download=False)


train_data,test_data = imdb['train'],imdb['test']

train_text = []
train_label = []

test_text = []
test_label = []

for text,label in train_data:
    train_text.append(text.numpy().decode('utf-8'))
    train_label.append(label.numpy())

for text,label in test_data:
    test_text.append(text.numpy().decode('utf-8'))
    test_label.append(label.numpy())

keras_nlp.tokenizers.compute_word_piece_vocabulary(
    train_text,
    vocabulary_size = 10000,
    reserved_tokens =["[PAD]","[UNK]"],
    vocabulary_output_file = 'imdb_vocab_subwords.txt'
)

subword_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary = './imdb_vocab_subwords.txt'
)

txt_dataset = tf.data.Dataset.from_tensor_slices(train_text)

train_sequences = txt_dataset.map(lambda x : subword_tokenizer(x))

padded_sequences = tf.keras.utils.pad_sequences(train_sequences,padding='post',maxlen=120)

train_label = tf.convert_to_tensor(train_label)

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(120,)),
        tf.keras.layers.Embedding(10000,16),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ]
)


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


#model train
model.fit(x=padded_sequences,y=train_label,epochs=10)