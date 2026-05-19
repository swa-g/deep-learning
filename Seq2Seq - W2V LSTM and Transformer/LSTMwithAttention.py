import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Layer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# ==========================================
# 1. DATA CONFIGURATION & PREPROCESSING
# ==========================================

FILE_PATH = "./por.txt"
NUM_EXAMPLES = 5000  # Adjust depending on hardware capacity
BATCH_SIZE = 64
EMBEDDING_DIM = 256
UNITS = 512
EPOCHS = 10


def preprocess_sentence(s):
    # Convert to lowercase and strip whitespaces
    s = s.lower().strip()
    # Add space between words and punctuation marks e.g., "he is a boy." -> "he is a boy ."
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    # Replace everything except (a-z, A-Z, ".", "?", "!", ",")
    # Keeping accented characters for Portuguese
    s = re.sub(r"[^a-zA-Z?.!,¿áéíóúâêôãõçÀÉÍÓÚÂÊÔÃÕÇ]+", " ", s)
    s = s.strip()
    # Adding a start and an end token to the sentence so that the model knows when to start and stop predicting.
    s = "<start> " + s + " <end>"
    return s


def load_dataset(path, num_examples):
    lines = open(path, encoding="utf-8").read().strip().split("\n")

    en_sentences = []
    pt_sentences = []

    for line in lines[:num_examples]:
        parts = line.split("\t")
        if len(parts) >= 2:
            # Source language: English, Target language: Portuguese
            # You can swap parts[1] and parts[0] if you want to translate Pt -> En
            en_sentences.append(preprocess_sentence(parts[0]))
            pt_sentences.append(preprocess_sentence(parts[1]))

    return en_sentences, pt_sentences


def tokenize(lang):
    # Filter none ensures we keep < and > for our tokens
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

print(f"Total English vocabulary size: {vocab_input_size}")
print(f"Total Portuguese vocabulary size: {vocab_target_size}")
print(f"Max length input: {max_length_input}, Max length target: {max_length_target}")

# Create tf.data Dataset
dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor))
dataset = dataset.shuffle(len(input_tensor)).batch(BATCH_SIZE, drop_remainder=True)

# ==========================================
# 2. MODEL DEFINITION (LSTM + ATTENTION)
# ==========================================


class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(
            enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        # For LSTM, we need both hidden state (h) and cell state (c)
        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        return output, [state_h, state_c]

    def initialize_hidden_state(self):
        return [
            tf.zeros((self.batch_sz, self.enc_units)),
            tf.zeros((self.batch_sz, self.enc_units)),
        ]


class BahdanauAttention(Layer):

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(
            dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fc = Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # hidden[0] contains state_h from the encoder/previous decoder step
        context_vector, attention_weights = self.attention(hidden[0], enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the LSTM
        output, state_h, state_c = self.lstm(x, initial_state=hidden)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # x shape == (batch_size, vocab)
        x = self.fc(output)

        return x, [state_h, state_c], attention_weights


# Initialize Models
encoder = Encoder(vocab_input_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
decoder = Decoder(vocab_target_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)

# ==========================================
# 3. LOSS & OPTIMIZATION
# ==========================================

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)


def loss_function(real, pred):
    # Mask loss so we do not compute losses on padding 0 tokens
    mask = tf.math.not_equal(real, 0)
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# ==========================================
# 4. TRAINING LOOP
# ==========================================


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(
            [target_tokenizer.word_index["<start>"]] * BATCH_SIZE, 1
        )

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = loss / int(targ.shape[1])
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


print("\nStarting Training...")
for epoch in range(EPOCHS):
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for batch, (inp, targ) in enumerate(dataset):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print(
                f"Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}"
            )

    print(f"Epoch {epoch + 1} Complete. Total Loss: {total_loss / (batch+1):.4f}\n")

# ==========================================
# 5. TRANSLATION / EVALUATION
# ==========================================


def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    inputs = [
        input_tokenizer.word_index[i]
        for i in sentence.split(" ")
        if i in input_tokenizer.word_index
    ]
    inputs = pad_sequences([inputs], maxlen=max_length_input, padding="post")
    inputs = tf.convert_to_tensor(inputs)

    result = ""

    hidden = [tf.zeros((1, UNITS)), tf.zeros((1, UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_tokenizer.word_index["<start>"]], 0)

    for t in range(max_length_target):
        predictions, dec_hidden, attention_weights = decoder(
            dec_input, dec_hidden, enc_out
        )

        predicted_id = tf.argmax(predictions[0]).numpy()

        if target_tokenizer.index_word[predicted_id] == "<end>":
            return result, sentence

        result += target_tokenizer.index_word[predicted_id] + " "

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence


def translate(sentence):
    result, sentence = evaluate(sentence)
    print(f"Input: {sentence}")
    print(f"Predicted translation: {result}\n")


# Test Translations
print("--- Testing Translation Capabilities ---")
translate("Go.")
translate("Run!")
translate("I love studying languages.")