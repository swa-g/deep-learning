import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Layer, LayerNormalization, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# ==========================================
# 1. DATA CONFIGURATION & PREPROCESSING
# ==========================================

FILE_PATH = "por.txt"
NUM_EXAMPLES = 5000  # Adjust depending on hardware capacity
BATCH_SIZE = 64
EMBEDDING_DIM = 128
NUM_HEADS = 4
FF_DIM = 512          # Hidden layer size in feed-forward network
NUM_LAYERS = 2        # Number of Encoder/Decoder layers
DROPOUT_RATE = 0.1
EPOCHS = 15

def preprocess_sentence(s):
    s = s.lower().strip()
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = re.sub(r"[^a-zA-Z?.!,¿áéíóúâêôãõçÀÉÍÓÚÂÊÔÃÕÇ]+", " ", s)
    s = s.strip()
    s = "<start> " + s + " <end>"
    return s

def load_dataset(path, num_examples):
    lines = open(path, encoding="utf-8").read().strip().split("\n")
    en_sentences, pt_sentences = [], []
    for line in lines[:num_examples]:
        parts = line.split("\t")
        if len(parts) >= 2:
            en_sentences.append(preprocess_sentence(parts[0]))
            pt_sentences.append(preprocess_sentence(parts[1]))
    return en_sentences, pt_sentences

def tokenize(lang):
    lang_tokenizer = Tokenizer(filters="")
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = pad_sequences(tensor, padding="post")
    return tensor, lang_tokenizer

print("Loading and preprocessing dataset...")
input_lang, target_lang = load_dataset(FILE_PATH, NUM_EXAMPLES)

input_tensor, input_tokenizer = tokenize(input_lang)
target_tensor, target_tokenizer = tokenize(target_lang)

max_length_input = input_tensor.shape[1]
max_length_target = target_tensor.shape[1]

vocab_input_size = len(input_tokenizer.word_index) + 1
vocab_target_size = len(target_tokenizer.word_index) + 1

print(f"English Vocab: {vocab_input_size} | Portuguese Vocab: {vocab_target_size}")
print(f"Max input len: {max_length_input} | Max output len: {max_length_target}")

dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor))
dataset = dataset.shuffle(len(input_tensor)).batch(BATCH_SIZE, drop_remainder=True)

# ==========================================
# 2. TRANSFORMER COMPONENTS (CUSTOM LAYERS)
# ==========================================

def positional_encoding(length, depth):
    depth = depth // 2
    positions = np.arange(length)[:, np.newaxis]  
    depths = np.arange(depth)[np.newaxis, :] / depth  
    
    angle_rates = 1 / (10000**depths)  
    angle_rads = positions * angle_rates  
    
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model, mask_zero=False)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class DecoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, x, enc_output):
        # FIX: We use use_causal_mask=True instead of passing manual tensor tracking
        attn1 = self.mha1(query=x, value=x, key=x, use_causal_mask=True)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)
        
        attn2 = self.mha2(query=out1, value=enc_output, key=enc_output)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        return self.layernorm3(out2 + ffn_output)

# ==========================================
# 3. FULL TRANSFORMER MODEL
# ==========================================

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.enc_embedding = PositionalEmbedding(vocab_size=input_vocab_size, d_model=d_model)
        self.dec_embedding = PositionalEmbedding(vocab_size=target_vocab_size, d_model=d_model)
        
        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        
        self.final_layer = Dense(target_vocab_size)
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs):
        inp, tar = inputs

        # Process Encoder
        x = self.enc_embedding(inp)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x)
        enc_output = x

        # Process Decoder
        y = self.dec_embedding(tar)
        y = self.dropout(y)
        for layer in self.decoder_layers:
            y = layer(y, enc_output)
        
        logits = self.final_layer(y)
        return logits

# Initialize Model
transformer = Transformer(
    num_layers=NUM_LAYERS, d_model=EMBEDDING_DIM, num_heads=NUM_HEADS,
    dff=FF_DIM, input_vocab_size=vocab_input_size, target_vocab_size=vocab_target_size,
    dropout_rate=DROPOUT_RATE
)

# ==========================================
# 4. TRAINING SETUP WITH MASKED LOSS
# ==========================================

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.not_equal(real, 0)
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

@tf.function
def train_step(inp, targ):
    tar_inp = targ[:, :-1]
    tar_real = targ[:, 1:]

    with tf.GradientTape() as tape:
        predictions = transformer([inp, tar_inp], training=True)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    return loss

print("\nStarting Transformer Training...")
for epoch in range(EPOCHS):
    total_loss = 0
    for batch, (inp, targ) in enumerate(dataset):
        batch_loss = train_step(inp, targ)
        total_loss += batch_loss

        if batch % 100 == 0:
            print(f"Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}")

    print(f"Epoch {epoch + 1} Complete. Mean Loss: {total_loss / (batch+1):.4f}\n")

# ==========================================
# 5. TRANSLATION / INFERENCE EVALUATION
# ==========================================

def translate(sentence):
    cleaned_sentence = preprocess_sentence(sentence)
    inputs = [input_tokenizer.word_index[i] for i in cleaned_sentence.split(' ') if i in input_tokenizer.word_index]
    inputs = pad_sequences([inputs], maxlen=max_length_input, padding='post')
    encoder_input = tf.convert_to_tensor(inputs)

    start_index = target_tokenizer.word_index['<start>']
    end_index = target_tokenizer.word_index['<end>']
    output_tokens = tf.convert_to_tensor([[start_index]], dtype=tf.int32)

    for _ in range(max_length_target):
        predictions = transformer([encoder_input, output_tokens], training=False)
        prediction = predictions[:, -1, :]
        predicted_id = tf.argmax(prediction, axis=-1, output_type=tf.int32)[0]

        if predicted_id == end_index:
            break

        output_tokens = tf.concat([output_tokens, tf.expand_dims([predicted_id], 0)], axis=-1)

    translated_tokens = output_tokens.numpy()[0][1:] 
    result = " ".join([target_tokenizer.index_word[i] for i in translated_tokens if i in target_tokenizer.index_word])
    
    print(f"Input: {sentence}")
    print(f"Predicted translation: {result}\n")

print("--- Testing Transformer Translation Capabilities ---")
translate("Go.")
translate("Run!")
translate("I love studying languages.")