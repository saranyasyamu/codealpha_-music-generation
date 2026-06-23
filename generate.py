import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from music21 import note, chord, stream
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_music():
    try:
        # Load notes
        with open(os.path.join(BASE_DIR, "notes.pkl"), "rb") as f:
            notes = pickle.load(f)

        pitchnames = sorted(set(notes))
        n_vocab = len(pitchnames)

        note_to_int = dict((n, number) for number, n in enumerate(pitchnames))
        int_to_note = dict((number, n) for number, n in enumerate(pitchnames))

        sequence_length = 50

        # Load trained model
        model = load_model(os.path.join(BASE_DIR, "music_model.keras"))

        # Random starting sequence
        start = np.random.randint(0, len(notes) - sequence_length - 1)

        pattern = [note_to_int[n] for n in notes[start:start + sequence_length]]

        prediction_output = []

        # Generate 200 notes
        for i in range(200):

            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)

            prediction = model.predict(prediction_input, verbose=0)

            index = np.argmax(prediction)

            result = int_to_note[index]
            prediction_output.append(result)

            pattern.append(index)
            pattern = pattern[1:]

        # Convert to MIDI
        offset = 0
        output_notes = []

        for pattern in prediction_output:

            if "." in pattern or str(pattern).isdigit():
                notes_in_chord = str(pattern).split(".")
                chord_notes = []

                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.offset = offset
                    new_note.storedInstrument = None
                    chord_notes.append(new_note)

                new_chord = chord.Chord(chord_notes)
                output_notes.append(new_chord)

            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = None
                output_notes.append(new_note)

            offset += 0.5

        midi_stream = stream.Stream(output_notes)

        filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mid"
        output_path = os.path.join(BASE_DIR, filename)

        midi_stream.write("midi", fp=output_path)

        print("Music generated successfully!")
        print(f"Saved as {output_path}")

        return output_path

    except Exception as e:
        print("ERROR:", e)
        return None


if __name__ == "__main__":
    result = generate_music()
    print("Output file:", result)