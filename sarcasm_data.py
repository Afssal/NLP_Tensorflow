import json
import tensorflow_datasets as tfds
import tensorflow as tf

data = []
label = []
text = []
train_size = 70

with open('Sarcasm_Headlines_Dataset_v2.json','r') as file:
    for line in file:
        data.append(json.loads(line))


for line in data:
    label.append(line['is_sarcastic'])
    text.append(line['headline'])

train_data = text[:int((70/100*len(text)))]
train_label = label[:int((70/100*len(text)))]
test_data = text[int((70/100*len(text))):]
test_label = label[int((70/100*len(text))):]

print(len(train_data))
print(len(test_data))
# print(len(text))
# print(int((70/100*len(text))))
# print(data)

vectorize_layer = tf.keras.layers.TextVectorization(max_tokens = 10000)

vectorize_layer.adapt(text)

train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
train_sequences = train_dataset.map(lambda x : vectorize_layer(x))
train_padded = tf.keras.utils.pad_sequences(train_sequences,padding='post',maxlen=120)

train_label = tf.convert_to_tensor(train_label)

test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
test_sequences = test_dataset.map(lambda x : vectorize_layer(x))
test_padded = tf.keras.utils.pad_sequences(test_sequences,padding='post',maxlen=120)

test_label = tf.convert_to_tensor(test_label)

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(120,),
        tf.keras.layers.Embedding(10000,16),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid'))
    ]
)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x=train_padded,y=train_label,validation_data=(test_padded,test_label),epochs=10)