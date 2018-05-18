import glob
import numpy as np
import gc
from music21 import converter, instrument, note, chord, midi


class MidiUtils:
    midiPartsList = []

    def __init__(self, midi_file_path, minNote, maxNote, sequence_length, time_step, music_name):
        self.midi_file_path = midi_file_path
        self.minNote = int(minNote)
        self.maxNote = int(maxNote)
        self.range = maxNote - minNote
        self.sequence_length = sequence_length
        self.time_step = time_step
        self.music_name = music_name

    def preprocessing(self):
        seqList = []
        self.get_midi_data()

        for item in self.midiPartsList:
            nd = self.get_notes(item)
            matrix_length = int(item.duration.quarterLength / self.time_step)
            sq = self.get_matrix(nd, matrix_length)
            seqList.append(sq)

        sequence = self.sequence_join(seqList)
        print("Sequence Created. Size: ", sequence.shape[0], sequence.shape[1])
        network_input, network_output = self.IO_create(sequence)
        return network_input, network_output

    def postprocessing_get_midi(self, path):
        seqList = []
        self.get_midi_file(path)

        for item in self.midiPartsList:
            nd = self.get_notes(item)
            matrix_length = int(item.duration.quarterLength / self.time_step)
            sq = self.get_matrix(nd, matrix_length)
            seqList.append(sq)

        sequence = self.sequence_join(seqList)
        print("Sequence Created. Size: ", sequence.shape[0], sequence.shape[1])
        network_input, network_output = self.IO_create(sequence)
        return network_input, network_output

    def get_midi_data(self):
        self.midiPartsList.clear()
        count = 0
        for file in glob.glob(self.midi_file_path):
            midifile = converter.parse(file)
            parts = instrument.partitionByInstrument(midifile)
            self.midiPartsList.append(parts)
            count += 1
        print('Got all midi file: ', count)

    def get_notes(self, midi_file):
        notes = []
        chords = []
        notesDict = {n: [] for n in range(self.minNote, self.maxNote)}
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi_file)

        if parts:  # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(element)
            elif isinstance(element, chord.Chord):
                chords.append(element)

        for element in notes:
            if isinstance(element, note.Note):
                if notesDict.__contains__(element.pitch.midi):
                    notesDict[element.pitch.midi] += [[element.offset, element.offset + element.duration.quarterLength]]
        for element in chords:
            if isinstance(element, chord.Chord):
                for item in element:
                    if notesDict.__contains__(item.pitch.midi):
                        notesDict[item.pitch.midi] += [[element.offset, element.offset + element.duration.quarterLength]]

        return notesDict

    def get_matrix(self, notes_dict, time_total):
        sequence = np.zeros((time_total, len(notes_dict)))

        for element in notes_dict:
            for (start, end) in notes_dict[element]:
                start_t = (int)(start / self.time_step)
                end_t = (int)(end / self.time_step)
                sequence[start_t: end_t, element - self.minNote] = 1

        return sequence

    def sequence_join(self, sequence_list):
        return np.concatenate(sequence_list, axis=0)

    def IO_create(self, matrix):
        gc.collect()
        network_input = np.zeros((matrix.shape[0] - self.sequence_length, self.sequence_length, self.range))
        network_output = np.zeros((matrix.shape[0] - self.sequence_length, self.range))
        for i in range(0, matrix.shape[0] - self.sequence_length, 1):
            network_input[i, :, :] = matrix[i:i + self.sequence_length, :]
            network_output[i, :] = matrix[i + self.sequence_length, :]
        return network_input, network_output
