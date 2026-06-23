import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Load notes
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

print("Total notes:", len(notes))

# Create vocabulary
pitchnames = sorted(set(notes))
n_vocab = len(pitchnames)

print("Unique notes:", n_vocab)

note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

sequence_length = 50

network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]

    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)

print("Training patterns:", n_patterns)

network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(n_vocab)

network_output = to_categorical(network_output)

model = Sequential()

model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2])))
model.add(Dropout(0.3))
model.add(Dense(256, activation="relu"))
model.add(Dense(n_vocab, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam")

checkpoint = ModelCheckpoint(
    "music_model.keras",
    monitor="loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

print("Training started...")

model.fit(
    network_input,
    network_output,
    epochs=10,
    batch_size=64,
    callbacks=[checkpoint]
)

print("Training Completed!")