from music21 import converter, instrument, note, chord
import glob
import pickle

notes = []

midi_files = glob.glob("maestro-v3.0.0-midi/**/*.midi", recursive=True)

# Use only first 100 files for faster processing
midi_files = midi_files[:100]

print("Reading MIDI files...")
print("Total MIDI files:", len(midi_files))

for i, file in enumerate(midi_files):
    print(f"Processing {i+1}/{len(midi_files)}: {file}")

    try:
        midi = converter.parse(file)

        parts = instrument.partitionByInstrument(midi)

        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flatten().notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    except Exception as e:
        print("Skipped:", file)

with open("notes.pkl", "wb") as f:
    pickle.dump(notes, f)

print("Done!")
print("Total Notes:", len(notes))