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

# =============================================================================
# UTILITY FUNCTIONS
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

def apply_microtonal_shift(freq, shift_cents=0):
    return freq * (2 ** (shift_cents / 1200.0))

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
# ACOUSTIC PIANO EMULATION (SIMPLE)
# =============================================================================

def generate_acoustic_piano_sample(base_freq=220.0, dur=2.0):
    """
    A simple 'acoustic piano'-like envelope. Real sampling is best, 
    but here's a minimalistic approach:
    - multiple sine partials
    - exponential decay
    - a quick 'attack'
    """
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, endpoint=False)

    # Just a quick emulation
    partials = [
        (1.0, 1.0),  # (amplitude, multiple_of_base_freq)
        (0.7, 2.0),
        (0.3, 3.0)
    ]
    envelope = np.exp(-3.0 * t)  # fairly quick decay
    # add a short strong attack
    attack_len = int(0.01 * n)   # 10ms
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
# CHORD / HARMONY LOGIC
# =============================================================================

CHORD_INTERVALS = {
    "maj":  ([0, 4, 7], 1.0),
    "min":  ([0, 3, 7], 1.0),
    "dim":  ([0, 3, 6], 0.5),
    "aug":  ([0, 4, 8], 0.5),
    "7":    ([0, 4, 7, 10], 0.8),
    "maj7": ([0, 4, 7, 11], 0.8),
    "min7": ([0, 3, 7, 10], 0.8),
    "sus2": ([0, 2, 7], 0.9),
    "sus4": ([0, 5, 7], 0.9),
    "9":    ([0, 4, 7, 10, 14], 0.5),
    "11":   ([0, 4, 7, 10, 14, 17], 0.4),
    "13":   ([0, 4, 7, 10, 14, 17, 21], 0.4)
}

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]


class KeySignature:
    def __init__(self, root_midi=60, mode="major", microtonal_offset=0):
        self.root_midi = root_midi
        self.mode = mode
        self.microtonal_offset = microtonal_offset

    def get_scale_intervals(self):
        if self.mode == "major":
            return [self.root_midi + x for x in MAJOR_SCALE]
        else:
            return [self.root_midi + x for x in MINOR_SCALE]


def chord_name_to_frequencies(root_midi, chord_type, global_microtonal_offset=0, microtonal_chance=0.03):
    intervals, base_weight = CHORD_INTERVALS[chord_type]
    freqs = []
    for st in intervals:
        note_midi = root_midi + st
        f = midi_note_to_freq(note_midi)
        if global_microtonal_offset != 0:
            f = apply_microtonal_shift(f, global_microtonal_offset)
        # mild local shift
        if random.random() < microtonal_chance:
            shift_cents = random.randint(-10, 10)
            f = apply_microtonal_shift(f, shift_cents)
        freqs.append(f)
    return freqs


# =============================================================================
# CHORD PROGRESSION
# =============================================================================

def generate_chord_candidates(key_sig, chord_history):
    scale = key_sig.get_scale_intervals()
    chord_types = list(CHORD_INTERVALS.keys())
    candidates = []
    for root in scale:
        ctype = random.choice(chord_types)
        intervals, cweight = CHORD_INTERVALS[ctype]
        base_weight = cweight
        if chord_history:
            # naive V->I weighting
            last_root = chord_history[-1]["midi_root"]
            if (last_root % 12) == ((key_sig.root_midi + 7) % 12):
                if (root % 12) == (key_sig.root_midi % 12):
                    base_weight += 0.8
        cand = {
            "midi_root": root,
            "chord_type": ctype,
            "weight": base_weight
        }
        candidates.append(cand)

    # Some random outside chords
    for _ in range(5):
        outside_root = random.randint(48, 72)
        ctype = random.choice(chord_types)
        intervals, cweight = CHORD_INTERVALS[ctype]
        cand = {
            "midi_root": outside_root,
            "chord_type": ctype,
            "weight": 0.2 * cweight
        }
        candidates.append(cand)

    return candidates

def pick_weighted_chord(chord_candidates):
    weights = [c["weight"] for c in chord_candidates]
    total_weight = sum(weights)
    if total_weight <= 0:
        return random.choice(chord_candidates)
    r = random.random() * total_weight
    cum = 0
    for cand in chord_candidates:
        cum += cand["weight"]
        if r <= cum:
            return cand
    return chord_candidates[-1]

def chord_to_frequencies(chord_dict, key_sig):
    return chord_name_to_frequencies(
        chord_dict["midi_root"],
        chord_dict["chord_type"],
        global_microtonal_offset=key_sig.microtonal_offset,
        microtonal_chance=0.02
    )


# =============================================================================
# MELODY LIBRARY (Motifs) & MELODY MANAGER
# =============================================================================

# Each motif: a list of (time_in_beats, scale_degree_offset, duration_in_beats).
# We'll adapt these to the current chord/key by transposing scale degrees
# to chord tones or scale tones. We'll also try to adjust the last note
# to "lead" into the next chord's root or chord tone.

MELODY_LIBRARY = [
    # Very simple motifs, mostly one or two notes
    [(0.0, 0, 1.0)],        # One note at root
    [(0.0, 0, 0.5), (1.0, 2, 0.5)],
    [(0.0, 0, 1.0), (2.0, 1, 1.0)],
    # A slightly longer motif
    [(0.0, 0, 0.5), (1.0, 2, 0.5), (2.0, 4, 1.0)]
]

class MelodyManager:
    """
    Similar to the chord progression manager, but for melodic motifs.
    We pick or repeat motifs for each "section," adapt them to chord changes,
    and aim to lead into the next chord.
    """
    def __init__(self, key_sig):
        self.key_sig = key_sig
        self.motif_history = []
        self.current_motif = []
        self.current_motif_index = 0

    def pick_new_motif(self):
        # Possibly revisit a previous motif or pick from library
        if self.motif_history and random.random() < 0.4:
            motif = random.choice(self.motif_history)
        else:
            motif = random.choice(MELODY_LIBRARY)
        self.motif_history.append(motif)
        self.current_motif = motif
        self.current_motif_index = 0

    def get_notes_for_measure(self, chord_freqs, next_chord_root=None):
        """
        Return a list of notes (time_in_beats, frequency, duration_in_beats)
        based on the current motif. 
        We'll adapt the motif's scale degree offsets to chord_freqs (or key scale).
        We'll also try to tweak the last note if next_chord_root is known.
        """
        if not chord_freqs:
            return []

        # If we have no motif or finished the last motif pattern, pick a new one
        if not self.current_motif or self.current_motif_index >= len(self.current_motif):
            self.pick_new_motif()

        notes_out = []
        while (self.current_motif_index < len(self.current_motif)):
            (start_beat, scale_offset, dur_beats) = self.current_motif[self.current_motif_index]
            # We'll map scale_offset to a chord tone or scale tone:
            freq = pick_melodic_tone(chord_freqs, self.key_sig, scale_offset)

            # If this is the last note of the motif, we might lead into next chord root
            if self.current_motif_index == len(self.current_motif) - 1 and next_chord_root:
                # Shift freq slightly closer to next_chord_root
                freq = lead_into_next_chord(freq, next_chord_root)

            notes_out.append((start_beat, freq, dur_beats))
            self.current_motif_index += 1

        return notes_out


def pick_melodic_tone(chord_freqs, key_sig, scale_offset):
    """
    Map the scale_offset (e.g. 0=chord root, 1=next chord tone, etc.) 
    to a note frequency in chord_freqs or the key's scale.
    """
    # We'll do a naive approach: chord_freqs are sorted from root -> 
    # but they might not be strictly root->3rd->5th->7th in ascending freq. 
    # We'll just pick chord_freqs[0] as "root", chord_freqs[1] as "next chord tone", etc. 
    # If we exceed chord_freqs length, fall back to the key scale.
    idx = scale_offset
    if idx < len(chord_freqs) and idx >= 0:
        return chord_freqs[idx]
    else:
        # fallback to key scale
        scale_midi = key_sig.get_scale_intervals()
        note_midi = random.choice(scale_midi)
        return midi_note_to_freq(note_midi)

def lead_into_next_chord(current_freq, next_chord_root):
    """
    Nudges current_freq toward next_chord_root to create a leading tone effect.
    We can do a simple average or small shift.
    """
    # a simple approach: shift halfway toward next chord root freq
    lead_freq = 0.7 * current_freq + 0.3 * next_chord_root
    return lead_freq


def generate_acoustic_piano_note(freq, dur=0.5):
    """
    Simple approach: pitch-shift a 220Hz "acoustic piano" sample to freq.
    """
    # We'll create a short sample for each note event:
    base_wave = generate_acoustic_piano_sample(220.0, dur=2.0)  # 2s sample
    pitched = pitch_shift(base_wave, 220.0, freq, dur)
    # Slight volume
    return pitched * 0.3


def render_melody_line(notes, measure_len):
    """
    Render the given notes (time_in_beats, freq, duration_in_beats) 
    into an audio buffer of length measure_len (samples).
    """
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
        self.chord_history = []

        self.current_section_length = random.randint(MIN_SECTION_LENGTH, MAX_SECTION_LENGTH)
        self.current_section_chords = []
        self.current_section_index = 0
        self.measure_count_in_section = 0
        self.previous_sections = []

        # Melody manager
        self.melody_manager = MelodyManager(initial_key)

        # Generate the first section
        self.start_new_section()

    def start_new_section(self):
        if random.random() < 0.15:
            self.modulate_key()

        chord_candidates = generate_chord_candidates(self.current_key, self.chord_history)
        new_chords = []
        num_chords = random.choice([4, 4, 5])
        for _ in range(num_chords):
            cand = pick_weighted_chord(chord_candidates)
            new_chords.append(cand)

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
        old_root = self.current_key.root_midi
        new_root = old_root + random.choice([-5, -2, 2, 5, 7])
        new_mode = random.choice(["major", "minor"])
        micro_offset = random.choice([0, -10, 10])
        self.current_key = KeySignature(new_root, new_mode, micro_offset)
        print(f"*** Modulated Key -> {self.pretty_key_name(self.current_key)} ***")

    def get_next_chord(self):
        if self.measure_count_in_section >= self.current_section_length:
            self.start_new_section()

        chord_dict = self.current_section_chords[self.current_section_index]
        freqs = chord_to_frequencies(chord_dict, self.current_key)
        self.chord_history.append({
            "midi_root": chord_dict["midi_root"],
            "chord_type": chord_dict["chord_type"]
        })

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

    def pretty_key_name(self, key_sig):
        note_names = ["C", "C#", "D", "D#", "E", "F",
                      "F#", "G", "G#", "A", "A#", "B"]
        root = key_sig.root_midi % 12
        return f"{note_names[root]} {key_sig.mode} (micro={key_sig.microtonal_offset}c)"


# =============================================================================
# BASS
# =============================================================================

def generate_bass_line(chord_freqs, measure_len_samps, key_sig):
    out = np.zeros(measure_len_samps, dtype=np.float32)
    if not chord_freqs:
        return out

    root_freq = chord_freqs[0]
    alt_freq = chord_freqs[1] if len(chord_freqs) > 1 else root_freq

    beats = [0.0, 2.0]
    for beat in beats:
        freq = root_freq if beat == 0.0 else alt_freq
        start_idx = beats_to_samples(beat, BPM, SAMPLE_RATE)
        bass_wave = generate_simple_bass(freq, 0.4)
        end_idx = min(start_idx + len(bass_wave), measure_len_samps)
        out[start_idx:end_idx] += bass_wave[:end_idx-start_idx]

    return out

def generate_simple_bass(freq, dur=0.4):
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    envelope = np.exp(-4 * t)
    return (wave * envelope * 0.4).astype(np.float32)


# =============================================================================
# DRUMS
# =============================================================================

def generate_drums(measure_len, measure_index):
    out = np.zeros(measure_len, dtype=np.float32)
    events = []
    events.append((0.0, 'kick'))
    if random.random() < 0.5:
        events.append((random.choice([1.5, 2, 2.5, 3]), 'kick'))
    events.append((1.98, 'snare'))
    events.append((3.98, 'snare'))

    for i in range(BEATS_PER_MEASURE * 2):
        hat_time = i * 0.5
        if random.random() < 0.85:
            events.append((hat_time, 'hat'))

    if (measure_index % 8) == 0 and random.random() < 0.3:
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
        out[start_idx:end_idx] += wave[:end_idx - start_idx]

    return out


# =============================================================================
# COMPOSITE MEASURE
# =============================================================================

def generate_one_measure(chord_prog_mgr):
    measure_len = int(MEASURE_SECONDS * SAMPLE_RATE)

    chord_freqs, chord_dict = chord_prog_mgr.get_next_chord()

    # Next chord's root freq (for melody leading). 
    next_chord_index = (chord_prog_mgr.current_section_index) % len(chord_prog_mgr.current_section_chords)
    next_chord_dict = chord_prog_mgr.current_section_chords[next_chord_index]
    next_chord_root = midi_note_to_freq(next_chord_dict["midi_root"])

    # chord wave
    chord_wave = generate_chord(chord_freqs, dur=MEASURE_SECONDS, amplitude=0.3)

    # drums
    drum_wave = generate_drums(measure_len, chord_prog_mgr.measure_count)

    # bass
    bass_wave = generate_bass_line(chord_freqs, measure_len, chord_prog_mgr.current_key)

    # melody
    # obtain a measure's worth of notes from melody manager
    melody_notes = chord_prog_mgr.melody_manager.get_notes_for_measure(chord_freqs, next_chord_root)
    melody_wave = render_melody_line(melody_notes, measure_len)

    combined = chord_wave + drum_wave + bass_wave + melody_wave

    # filter, noise, normalize
    filtered = low_pass_filter(combined, cutoff_freq=5000)
    crackle = add_vinyl_noise(len(filtered), noise_level=0.02)
    result = filtered + crackle
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result /= peak
    result *= db_to_linear(DECIBEL_HEADROOM)

    return result.astype(np.float32)


def generate_chord(chord_freqs, dur=1.0, amplitude=0.3):
    """
    We'll use our acoustic piano sample at 220Hz as the base, then pitch-shift 
    for each chord tone.
    """
    n = int(dur * SAMPLE_RATE)
    chord_wave = np.zeros(n, dtype=np.float32)

    # We'll build a single 'base' piano sample once (2s), reuse it.
    base_piano = generate_acoustic_piano_sample(220.0, 2.0)

    for f in chord_freqs:
        pitched = pitch_shift(base_piano, 220.0, f, dur)
        chord_wave += pitched * amplitude

    # Fade in/out
    attack_len = int(0.1 * n)
    decay_len = int(0.2 * n)
    for i in range(attack_len):
        chord_wave[i] *= (i / attack_len)
    for i in range(decay_len):
        chord_wave[-1 - i] *= (i / decay_len)

    return chord_wave


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

        outdata[:, 0] = out  # mono
    except Exception as e:
        print("Exception in callback:", e)
        outdata.fill(0)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Lo-fi jam with simpler acoustic piano-style melodies from a motif library...")

    initial_key = KeySignature(root_midi=60, mode="major", microtonal_offset=0)  # C major
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
