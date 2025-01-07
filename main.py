import numpy as np
import random
import math
import queue
import threading
import time

import sounddevice as sd
from scipy.signal import butter, filtfilt

# =============================================================================
# CONFIG
# =============================================================================

SAMPLE_RATE = 44100
BPM = 70
BEATS_PER_MEASURE = 4
MEASURE_SECONDS = BEATS_PER_MEASURE * (60.0 / BPM)

CHUNK_SIZE = 2048
MAX_QUEUE_SIZE = 16

DECIBEL_HEADROOM = -3.0

MIN_SECTION_LENGTH = 10
MAX_SECTION_LENGTH = 14

# You can shift chord & melody octaves if desired
CHORD_OCTAVE_SHIFT = -12
MELODY_OCTAVE_SHIFT = -12

# We'll drastically reduce the chance of dim/aug
# and keep them from repeating measure to measure
DIM_OR_AUG_RARE_WEIGHT = 0.05

# We'll store the chord types we want to *primarily* focus on
PREFERRED_CHORD_TYPES = [
    "maj", "min", "7", "maj7", "min7",
    # We'll keep sus2, sus4, 9, 11, 13, etc. but at lower weights
    "sus2", "sus4", "9", "11", "13"
    # dim/aug are either omitted or used rarely
]

CHORD_RHYTHM_PATTERNS = [
    [(0.0, 1.0, 1.0), (2.0, 1.0, 0.8)],
    [(0.0, 0.5, 1.0), (1.5, 0.5, 0.7), (2.5, 0.5, 0.7)],
    [(0.0, 1.5, 1.0), (2.5, 0.5, 1.2)],
]

# =============================================================================
# UTILITIES
# =============================================================================

def db_to_linear(db):
    return 10 ** (db / 20.0)

def low_pass_filter(signal, cutoff_freq, sample_rate=SAMPLE_RATE):
    b, a = butter(N=2, Wn=cutoff_freq/(0.5 * sample_rate), btype='low')
    return filtfilt(b, a, signal)

def add_vinyl_noise(num_samples, noise_level=0.02):
    noise = np.zeros(num_samples, dtype=np.float32)
    mask = np.random.choice([0, 1], size=num_samples, p=[0.98, 0.02])
    vals = (np.random.rand(num_samples) - 0.5) * 2
    noise = vals * mask * noise_level
    return noise

def midi_note_to_freq(midi_note):
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def beats_to_samples(num_beats, bpm=BPM, sr=SAMPLE_RATE):
    sec_per_beat = 60.0 / bpm
    return int(num_beats * sec_per_beat * sr)


# =============================================================================
# DRUMS
# =============================================================================

def generate_kick(dur=0.4):
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, endpoint=False)
    freq_sweep = np.linspace(100, 40, n)
    wave = np.sin(2*np.pi * freq_sweep * t)
    env = np.exp(-5 * t)
    return wave * env

def generate_snare(dur=0.2):
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, endpoint=False)
    noise = np.random.randn(n) * 0.5
    body = 0.5 * np.sin(2*np.pi * 200 * t)
    env = np.exp(-10 * t)
    return (noise + body) * env

def generate_hat(dur=0.1):
    n = int(dur * SAMPLE_RATE)
    noise = np.random.randn(n)
    filtered = noise - np.convolve(noise, np.ones(5)/5, mode='same')
    env = np.exp(-20 * np.linspace(0, dur, n))
    return filtered * env * 0.3


# =============================================================================
# ACOUSTIC PIANO (SIMPLE)
# =============================================================================

def generate_acoustic_piano_sample(base_freq=220.0, dur=2.0):
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, endpoint=False)

    partials = [
        (1.0, 1.0),
        (0.7, 2.0),
        (0.3, 3.0)
    ]
    envelope = np.exp(-3.0 * t)

    attack_len = int(0.01 * n)
    for i in range(attack_len):
        envelope[i] *= (i / attack_len * 0.8 + 0.2)

    wave = np.zeros(n, dtype=np.float32)
    for (amp, mul) in partials:
        wave += amp * np.sin(2 * np.pi * base_freq * mul * t)
    wave *= envelope
    return wave.astype(np.float32)

def pitch_shift(base_wave, base_freq, target_freq, dur):
    wave_len = len(base_wave)
    target_len = int(dur * SAMPLE_RATE)
    ratio = target_freq / base_freq
    base_indices = np.linspace(0, wave_len - 1, target_len) * ratio
    base_indices = np.clip(base_indices, 0, wave_len - 1)
    resampled = np.interp(base_indices, np.arange(wave_len), base_wave)
    return resampled.astype(np.float32)


# =============================================================================
# CHORD LOGIC
# =============================================================================

CHORD_INTERVALS = {
    "maj":  ([0, 4, 7], 1.0),
    "min":  ([0, 3, 7], 1.0),

    "7":    ([0, 4, 7, 10], 0.9),
    "maj7": ([0, 4, 7, 11], 0.9),
    "min7": ([0, 3, 7, 10], 0.9),

    # We'll keep these but at very low weighting:
    "dim":  ([0, 3, 6], DIM_OR_AUG_RARE_WEIGHT),
    "aug":  ([0, 4, 8], DIM_OR_AUG_RARE_WEIGHT),

    # We'll keep 9, 11, 13 but lower weighting than 7
    "9":    ([0, 4, 7, 10, 14], 0.4),
    "11":   ([0, 4, 7, 10, 14, 17], 0.3),
    "13":   ([0, 4, 7, 10, 14, 17, 21], 0.3),

    "sus2": ([0, 2, 7], 0.5),
    "sus4": ([0, 5, 7], 0.5),
}

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

class KeySignature:
    def __init__(self, root_midi=60, mode="major"):
        self.root_midi = root_midi
        self.mode = mode

    def get_scale_midi(self):
        if self.mode == "major":
            return [self.root_midi + x for x in MAJOR_SCALE]
        else:
            return [self.root_midi + x for x in MINOR_SCALE]

def chord_name_to_frequencies(root_midi, chord_type):
    """
    We'll basically skip microtonal offsets altogether for maximum consonance.
    We'll also shift the chord down an octave if desired.
    """
    intervals, _ = CHORD_INTERVALS[chord_type]
    base_root = (root_midi + CHORD_OCTAVE_SHIFT)
    freqs = []
    for st in intervals:
        note_midi = base_root + st
        freqs.append(midi_note_to_freq(note_midi))

    # Optionally add some extra octave notes (root, maybe 3rd)
    # to fill out the chord if it's strongly major/minor/7/etc.
    if chord_type in ["maj", "min", "7", "maj7", "min7", "9", "11", "13"]:
        # add a lower root
        if random.random() < 0.5:
            freqs.append(midi_note_to_freq(base_root - 12))
        # add a higher root or 3rd
        if random.random() < 0.5 and intervals:
            up_int = intervals[1] if len(intervals) > 1 else 0
            high_midi = base_root + up_int + 12
            freqs.append(midi_note_to_freq(high_midi))

    freqs.sort()
    return freqs


# =============================================================================
# CHORD PROGRESSION
# =============================================================================

def generate_chord_candidates(key_sig, chord_history):
    scale_midi = key_sig.get_scale_midi()

    # We'll only pick from our PREFERRED_CHORD_TYPES (major, minor, 7, etc.)
    # but we keep dim/aug in the dictionary if a random pick occurs
    chord_types = PREFERRED_CHORD_TYPES

    candidates = []
    for root in scale_midi:
        ctype = random.choice(chord_types)
        intervals, weight = CHORD_INTERVALS[ctype]
        base_weight = weight

        # Avoid repeating aug/dim if we just had one
        if chord_history:
            last_chord_type = chord_history[-1]["chord_type"]
            if last_chord_type in ["dim", "aug"] and ctype in ["dim", "aug"]:
                base_weight = 0.0  # effectively skip consecutive dim/aug

        # Increase weight if it's purely major or minor
        if ctype in ["maj", "min"]:
            base_weight += 0.3

        # Simple V->I weighting
        last_root = chord_history[-1]["midi_root"] if chord_history else None
        if last_root is not None:
            # if last chord root = key_root+7 => V
            if (last_root % 12) == ((key_sig.root_midi + 7) % 12):
                if (root % 12) == (key_sig.root_midi % 12):
                    base_weight += 1.0

        cand = {
            "midi_root": root,
            "chord_type": ctype,
            "weight": max(base_weight, 0.0)  # clamp to zero if negative
        }
        candidates.append(cand)

    # Might add a tiny subset of outside chords, but weigh them very low
    for _ in range(3):
        outside_root = random.randint(36, 72)  # somewhat wide range
        ctype = random.choice(chord_types)
        intervals, weight = CHORD_INTERVALS[ctype]
        # also avoid consecutive dim/aug outside
        if chord_history and chord_history[-1]["chord_type"] in ["dim", "aug"] and ctype in ["dim", "aug"]:
            base_weight = 0.0
        else:
            base_weight = 0.1 * weight
        candidates.append({
            "midi_root": outside_root,
            "chord_type": ctype,
            "weight": base_weight
        })

    return candidates

def pick_weighted_chord(chord_candidates):
    weights = [c["weight"] for c in chord_candidates]
    total_weight = sum(weights)
    if total_weight <= 0:
        # fallback if all weights are zero
        return random.choice(chord_candidates)
    r = random.random() * total_weight
    cum = 0
    for cand in chord_candidates:
        cum += cand["weight"]
        if r <= cum:
            return cand
    return chord_candidates[-1]


# =============================================================================
# MELODY LOGIC
# =============================================================================

# We'll keep a few short motifs, but we'll ensure the notes primarily line up with chord tones.
MELODY_LIBRARY = [
    # Each entry: (start_beat, "chord_tone_index", duration_in_beats)
    [(0.0, 0, 1.0)],  
    [(0.0, 0, 0.5), (1.5, 1, 0.5)],
    [(0.0, 0, 1.0), (2.0, 1, 1.0)],
    [(0.0, 0, 0.5), (2.0, 2, 1.5)],
]

class MelodyManager:
    def __init__(self, key_sig):
        self.key_sig = key_sig
        self.current_motif = []
        self.current_motif_index = 0
        self.motif_history = []

    def pick_new_motif(self):
        # Weighted random pick from MELODY_LIBRARY or from history
        if self.motif_history and random.random() < 0.4:
            motif = random.choice(self.motif_history)
        else:
            motif = random.choice(MELODY_LIBRARY)
        self.motif_history.append(motif)
        self.current_motif = motif
        self.current_motif_index = 0

    def get_notes_for_measure(self, chord_freqs, next_chord_root_freq):
        """
        We'll mostly pick chord tones for the melody. 
        chord_freqs are sorted. We'll also treat the top chord freq as "upper extension".
        next_chord_root_freq for mild leading if we want to shift the last note up or down slightly.
        """
        if not chord_freqs:
            return []

        if not self.current_motif or self.current_motif_index >= len(self.current_motif):
            self.pick_new_motif()

        notes = []
        while self.current_motif_index < len(self.current_motif):
            start_beat, chord_tone_idx, dur_beats = self.current_motif[self.current_motif_index]

            # clamp chord_tone_idx so it doesn't exceed chord_freqs
            if chord_tone_idx < 0: 
                chord_tone_idx = 0
            if chord_tone_idx >= len(chord_freqs):
                chord_tone_idx = len(chord_freqs) - 1

            freq = chord_freqs[chord_tone_idx]
            # shift melody an octave if needed
            freq *= 2 ** (MELODY_OCTAVE_SHIFT / 12.0)

            # maybe some small chance to pick a diatonic passing tone
            # but keep it subtle
            if random.random() < 0.1:
                scale_midi = self.key_sig.get_scale_midi()
                freq = midi_note_to_freq(random.choice(scale_midi) + MELODY_OCTAVE_SHIFT)

            # leading note if it's last in motif
            if self.current_motif_index == len(self.current_motif) - 1 and next_chord_root_freq:
                # nudge freq slightly toward next chord root
                freq = 0.8 * freq + 0.2 * (next_chord_root_freq * 2 ** (MELODY_OCTAVE_SHIFT/12.0))

            notes.append((start_beat, freq, dur_beats))
            self.current_motif_index += 1

        return notes

def generate_acoustic_piano_note(freq, dur=0.5):
    base_wave = generate_acoustic_piano_sample(220.0, dur=2.0)
    pitched = pitch_shift(base_wave, 220.0, freq, dur)
    return pitched * 0.3

def render_melody_line(notes, measure_len):
    out = np.zeros(measure_len, dtype=np.float32)
    for (start_beat, freq, dur_beats) in notes:
        start_idx = beats_to_samples(start_beat, BPM, SAMPLE_RATE)
        note_dur_sec = dur_beats * (60.0 / BPM)
        note_wave = generate_acoustic_piano_note(freq, note_dur_sec)

        end_idx = min(start_idx + len(note_wave), measure_len)
        out[start_idx:end_idx] += note_wave[:end_idx - start_idx]
    return out


# =============================================================================
# CHORD PROGRESSION MANAGER
# =============================================================================

class ChordProgressionManager:
    def __init__(self, initial_key):
        self.current_key = initial_key
        self.measure_count = 0
        self.current_section_chords = []
        self.current_section_index = 0
        self.measure_count_in_section = 0
        self.current_section_length = random.randint(MIN_SECTION_LENGTH, MAX_SECTION_LENGTH)
        self.chord_history = []
        self.previous_sections = []

        self.melody_manager = MelodyManager(initial_key)
        self.start_new_section()

    def start_new_section(self):
        # Possibly shift key by small amounts but let's keep it less frequent
        if random.random() < 0.1:
            self.modulate_key()

        chord_candidates = generate_chord_candidates(self.current_key, self.chord_history)
        new_chords = []
        # mostly pick 4-chord sections
        num_chords = random.choice([4,4,5])
        for _ in range(num_chords):
            cand = pick_weighted_chord(chord_candidates)
            new_chords.append(cand)

        # occasional reuse of old progression
        if self.previous_sections and random.random() < 0.2:
            new_chords = random.choice(self.previous_sections)

        self.current_section_chords = new_chords
        self.current_section_index = 0
        self.current_section_length = random.randint(MIN_SECTION_LENGTH, MAX_SECTION_LENGTH)
        self.measure_count_in_section = 0
        self.previous_sections.append(new_chords)

        print("=== NEW SECTION ===")
        for c in new_chords:
            print(f"   -> {self.pretty_chord_name(c)}")

    def modulate_key(self):
        # small transposition
        step = random.choice([-5, -2, 2, 5])
        new_root = self.current_key.root_midi + step
        new_mode = random.choice(["major", "minor"])
        self.current_key = KeySignature(new_root, new_mode)
        print(f"*** Modulated Key -> {self.pretty_key_name()} ***")

    def get_next_chord(self):
        if self.measure_count_in_section >= self.current_section_length:
            self.start_new_section()

        chord_dict = self.current_section_chords[self.current_section_index]
        freqs = chord_name_to_frequencies(chord_dict["midi_root"], chord_dict["chord_type"])

        self.chord_history.append(chord_dict)

        self.current_section_index = (self.current_section_index + 1) % len(self.current_section_chords)
        self.measure_count_in_section += 1
        self.measure_count += 1

        return freqs, chord_dict

    def pretty_chord_name(self, chord_dict):
        note_names = ["C", "C#", "D", "D#", "E", "F", 
                      "F#", "G", "G#", "A", "A#", "B"]
        root = chord_dict["midi_root"] % 12
        octave = chord_dict["midi_root"] // 12
        return f"{note_names[root]}{octave} {chord_dict['chord_type']}"

    def pretty_key_name(self):
        note_names = ["C", "C#", "D", "D#", "E", "F", 
                      "F#", "G", "G#", "A", "A#", "B"]
        root = self.current_key.root_midi % 12
        return f"{note_names[root]} {self.current_key.mode}"


# =============================================================================
# BASS
# =============================================================================

def generate_bass_line(chord_freqs, measure_len_samps):
    out = np.zeros(measure_len_samps, dtype=np.float32)
    if not chord_freqs:
        return out

    # pick the lowest chord freq for the root
    root_freq = chord_freqs[0]
    # maybe pick the next chord freq for a simple V or 3rd
    alt_freq = chord_freqs[1] if len(chord_freqs) > 1 else root_freq

    # simple pattern: beat 1 (root), beat 3 (alt)
    for beat in [0.0, 2.0]:
        freq = root_freq if beat == 0.0 else alt_freq
        # short sub-bass
        wave = generate_simple_bass(freq, 0.4)
        start_idx = beats_to_samples(beat, BPM, SAMPLE_RATE)
        end_idx = min(start_idx + len(wave), measure_len_samps)
        out[start_idx:end_idx] += wave[:end_idx-start_idx]

    return out

def generate_simple_bass(freq, dur=0.4):
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, endpoint=False)
    wave = np.sin(2*np.pi*freq*t)
    env = np.exp(-3 * t)
    return (wave * env * 0.4).astype(np.float32)


# =============================================================================
# DRUMS
# =============================================================================

def generate_drums(measure_len, measure_index):
    out = np.zeros(measure_len, dtype=np.float32)
    events = []
    events.append((0.0, 'kick'))
    # occasional extra kick
    if random.random() < 0.5:
        events.append((random.choice([1.5, 2, 2.5, 3]), 'kick'))
    events.append((1.98, 'snare'))
    events.append((3.98, 'snare'))

    for i in range(BEATS_PER_MEASURE*2):
        hat_time = i*0.5
        if random.random() < 0.85:
            events.append((hat_time, 'hat'))

    if (measure_index % 8) == 0 and random.random() < 0.3:
        # small fill
        events.append((3.5, 'snare'))
        events.append((3.75, 'snare'))

    events.sort(key=lambda x: x[0])
    beat_dur = 60.0 / BPM

    for (time_in_beats, instr) in events:
        start_sec = time_in_beats * beat_dur
        start_idx = int(start_sec * SAMPLE_RATE)
        if instr == 'kick':
            wave = generate_kick()
        elif instr == 'snare':
            wave = generate_snare()
        elif instr == 'hat':
            wave = generate_hat()
        else:
            wave = np.zeros(0, dtype=np.float32)
        end_idx = min(start_idx + len(wave), measure_len)
        out[start_idx:end_idx] += wave[:end_idx-start_idx]

    return out


# =============================================================================
# CHORD RHYTHM
# =============================================================================

def generate_chord_rhythm_line(chord_freqs, measure_len):
    out = np.zeros(measure_len, dtype=np.float32)
    if not chord_freqs:
        return out

    pattern = random.choice(CHORD_RHYTHM_PATTERNS)
    base_piano = generate_acoustic_piano_sample(220.0, 2.0)

    for (start_beat, dur_beats, amp_scale) in pattern:
        chord_dur_sec = dur_beats * (60.0 / BPM)
        chord_hit_wave = np.zeros(int(chord_dur_sec * SAMPLE_RATE), dtype=np.float32)
        for f in chord_freqs:
            partial = pitch_shift(base_piano, 220.0, f, chord_dur_sec)
            chord_hit_wave += partial * (0.3 * amp_scale)

        # short fade in/out
        a_len = int(0.02 * len(chord_hit_wave))
        d_len = int(0.1 * len(chord_hit_wave))
        for i in range(a_len):
            chord_hit_wave[i] *= (i / a_len)
        for i in range(d_len):
            chord_hit_wave[-1 - i] *= (i / d_len)

        start_idx = beats_to_samples(start_beat, BPM, SAMPLE_RATE)
        end_idx = min(start_idx + len(chord_hit_wave), measure_len)
        out[start_idx:end_idx] += chord_hit_wave[:end_idx - start_idx]

    return out


# =============================================================================
# COMPOSITE MEASURE
# =============================================================================

def generate_one_measure(chord_prog_mgr):
    measure_len = int(MEASURE_SECONDS * SAMPLE_RATE)

    chord_freqs, chord_dict = chord_prog_mgr.get_next_chord()
    # next chord root freq for melody lead
    next_index = chord_prog_mgr.current_section_index % len(chord_prog_mgr.current_section_chords)
    next_chord_dict = chord_prog_mgr.current_section_chords[next_index]
    next_chord_root_freq = midi_note_to_freq((next_chord_dict["midi_root"] + CHORD_OCTAVE_SHIFT))

    # 1) chord
    chord_wave = generate_chord_rhythm_line(chord_freqs, measure_len)

    # 2) drums
    drum_wave = generate_drums(measure_len, chord_prog_mgr.measure_count)

    # 3) bass
    bass_wave = generate_bass_line(chord_freqs, measure_len)

    # 4) melody
    melody_notes = chord_prog_mgr.melody_manager.get_notes_for_measure(chord_freqs, next_chord_root_freq)
    melody_wave = render_melody_line(melody_notes, measure_len)

    combined = chord_wave + drum_wave + bass_wave + melody_wave

    filtered = low_pass_filter(combined, cutoff_freq=5000)
    crackle = add_vinyl_noise(len(filtered), noise_level=0.02)
    result = filtered + crackle
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result /= peak
    result *= db_to_linear(DECIBEL_HEADROOM)

    return result.astype(np.float32)


# =============================================================================
# BACKGROUND PRODUCER THREAD
# =============================================================================

audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

def producer_thread(chord_prog_mgr):
    while True:
        measure_audio = generate_one_measure(chord_prog_mgr)
        idx = 0
        while idx < len(measure_audio):
            block = measure_audio[idx:idx+CHUNK_SIZE]
            audio_queue.put(block, block=True)
            idx += CHUNK_SIZE


# =============================================================================
# CALLBACK
# =============================================================================

def audio_callback(outdata, frames, time_info, status):
    if status:
        print("Status:", status)
    try:
        block = audio_queue.get(block=True)
        if len(block) < frames:
            out = np.zeros(frames, dtype=np.float32)
            out[:len(block)] = block
        else:
            out = block[:frames]
        outdata[:, 0] = out
    except Exception as e:
        print("Exception in callback:", e)
        outdata.fill(0)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Lo-fi jam starting...")

    initial_key = KeySignature(root_midi=48, mode="major")  # C3
    chord_mgr = ChordProgressionManager(initial_key)

    thread = threading.Thread(
        target=producer_thread,
        args=(chord_mgr,),
        daemon=True
    )
    thread.start()

    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        channels=1,
        dtype='float32',
        callback=audio_callback
    ):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting on Ctrl+C...")


if __name__ == '__main__':
    main()
