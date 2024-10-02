import tensorflow_datasets as tfds
import tensorflow as tf

imdb,info = tfds.load('imdb_reviews',with_info=True,as_supervised=True)

train_data,test_data = imdb['train'],imdb['test']

train_text = []
train_label = []

test_text = []
test_label = []

#adding train and test data and labels to list as numpy 
for text,label in train_data:
    train_text.append(text.numpy().decode('utf-8'))
    train_label.append(label.numpy())

for text,label in test_data:
    test_text.append(text.numpy().decode('utf-8'))
    test_label.append(label.numpy())

# print(train_label[1],train_text[0])

#initialze textvector module
vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=10000)

#create vocabulary
vectorize_layer.adapt(train_text)

#get vocabulary
vocabulary = vectorize_layer.get_vocabulary()

# print(vocabulary)

#create tensorflow dataset
txt_dataset = tf.data.Dataset.from_tensor_slices(train_text)

#map tensorflow dataset to vector
train_sequences = txt_dataset.map(lambda x : vectorize_layer(x))

#pad data
padded_sequence = tf.keras.utils.pad_sequences(train_sequences,padding='post',maxlen=120)

#convert label to tensor
train_label = tf.convert_to_tensor(train_label)

#test data
test_dataset = tf.data.Dataset.from_tensor_slices(test_text)

test_sequence = test_dataset.map(lambda x : vectorize_layer(x))

test_padded = tf.keras.utils.pad_sequences(test_sequence,padding='post',maxlen=120)

# print(padded_sequence)

test_label = tf.convert_to_tensor(test_label)

#create model
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
model.fit(x=padded_sequence,y=train_label,validation_data = (test_padded,test_label),epochs=10)


