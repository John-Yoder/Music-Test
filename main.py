import numpy as np
import random
import math
import queue
import threading
import time

import tkinter as tk
import sounddevice as sd
from scipy.signal import butter, filtfilt

# =============================================================================
# GLOBAL PARAMS / SHARED STATE
# =============================================================================

global_params = {
    "master_volume": 0.5,
    "piano_volume": 0.5,
    "bass_volume": 0.5,
    "drum_volume": 0.5,
    "noise_level": 0.5,
    "tempo": 70,
    "chord_dissonance": 0.5,
    "chord_extensions": 0.5,
    "drum_busyness": 0.5,
    # We'll default mode_cycle_speed to a small value (0.05) => very rare random mode changes
    "mode_cycle_speed": 0.05,
    "current_mode": "major",
    "key_root_midi": 48,
    # Synth volume for the new sweeping pad
    "synth_volume": 0.4,
}

def get_bpm():
    return global_params["tempo"]

def get_noise_amount():
    return 0.05 * global_params["noise_level"]

def get_master_volume():
    return global_params["master_volume"]

# =============================================================================
# SCROLLABLE UI CONTAINER
# =============================================================================

class ScrollableFrame(tk.Frame):
    """
    A utility that uses a canvas + scrollbar + an interior frame,
    so that we can place as many controls as we want and scroll through them.
    """
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.canvas = tk.Canvas(self, borderwidth=0)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.interior = tk.Frame(self.canvas)

        self.interior.bind(
            "<Configure>",
            lambda event: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.interior, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Layout
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")


# =============================================================================
# AUDIO CONFIG
# =============================================================================

SAMPLE_RATE = 44100
BEATS_PER_MEASURE = 4
CHUNK_SIZE = 2048
MAX_QUEUE_SIZE = 16
DECIBEL_HEADROOM = -3.0

def get_measure_seconds():
    bpm = get_bpm()
    return BEATS_PER_MEASURE * (60.0 / bpm)

def measure_len_samples():
    return int(get_measure_seconds() * SAMPLE_RATE)

def db_to_linear(db):
    return 10 ** (db / 20.0)

def beats_to_samples(num_beats):
    bpm = get_bpm()
    sec_per_beat = 60.0 / bpm
    return int(num_beats * sec_per_beat * SAMPLE_RATE)

def low_pass_filter(signal, cutoff_freq=5000):
    from scipy.signal import butter, filtfilt
    b, a = butter(N=2, Wn=cutoff_freq/(0.5 * SAMPLE_RATE), btype='low')
    return filtfilt(b, a, signal)

def add_vinyl_noise(num_samples):
    noise_level = get_noise_amount()
    if noise_level <= 0:
        return np.zeros(num_samples, dtype=np.float32)
    noise = np.zeros(num_samples, dtype=np.float32)
    mask = np.random.choice([0, 1], size=num_samples, p=[0.98, 0.02])
    vals = (np.random.rand(num_samples) - 0.5) * 2
    noise = vals * mask * noise_level
    return noise

def midi_note_to_freq(midi_note):
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


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

def place_drum_hit(buffer_out, instr, time_in_beats):
    start_idx = beats_to_samples(time_in_beats)
    if start_idx >= len(buffer_out):
        return  # avoid out-of-range

    wave = None
    if instr == 'kick':
        wave = generate_kick()
    elif instr == 'snare':
        wave = generate_snare()
    elif instr == 'hat':
        wave = generate_hat()
    else:
        return

    end_idx = min(start_idx + len(wave), len(buffer_out))
    buffer_out[start_idx:end_idx] += wave[:(end_idx - start_idx)]


# =============================================================================
# BASIC PIANO
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
    if base_freq == 0:
        ratio = 1.0
    else:
        ratio = target_freq / base_freq
    base_indices = np.linspace(0, wave_len - 1, target_len) * ratio
    base_indices = np.clip(base_indices, 0, wave_len - 1)
    resampled = np.interp(base_indices, np.arange(wave_len), base_wave)
    return resampled.astype(np.float32)

def generate_piano_note(freq, dur=1.0):
    base_wave = generate_acoustic_piano_sample(220.0, 2.0)
    pitched = pitch_shift(base_wave, 220.0, freq, dur)
    volume = global_params["piano_volume"]
    return pitched * volume


# =============================================================================
# BACKGROUND SWEEPING SYNTH
# =============================================================================

def generate_sweeping_synth_sample(base_freq=220.0, dur=4.0):
    """
    A slow, pad-like wave. We'll do a saw with slight detune and an envelope.
    """
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, endpoint=False)

    freq1 = base_freq
    freq2 = base_freq * 1.01  # slight detune
    phase1 = (freq1 * t) % 1.0
    phase2 = (freq2 * t) % 1.0
    saw1 = 2.0 * (phase1 - 0.5)
    saw2 = 2.0 * (phase2 - 0.5)
    wave = 0.5 * saw1 + 0.5 * saw2

    # slow fade in/out
    attack_time = dur * 0.2
    release_time = dur * 0.3
    env = np.ones(n)
    attack_samples = int(attack_time * SAMPLE_RATE)
    for i in range(attack_samples):
        env[i] = i / attack_samples
    release_samples = int(release_time * SAMPLE_RATE)
    for i in range(release_samples):
        env[-1 - i] *= (1.0 - i/release_samples)

    wave *= env * 0.3
    return wave.astype(np.float32)

class BackgroundSynthManager:
    def __init__(self):
        pass

    def generate_pad_for_chord(self, chord_freqs):
        """
        We'll pick the lowest freq as the base, pitch-shift a ~4s sample to it,
        then we'll place it for 1 measure. 
        """
        out = np.zeros(measure_len_samples(), dtype=np.float32)
        if not chord_freqs:
            return out

        # pick the root or average
        base_freq = chord_freqs[0]
        # generate a 4s sweeping sample at 220Hz, then pitch shift
        base_synth = generate_sweeping_synth_sample(220.0, dur=4.0)
        # pitch shift to measure length
        measure_dur = get_measure_seconds()
        pitched = pitch_shift(base_synth, 220.0, base_freq, measure_dur)

        # optional second partial an octave up
        if random.random() < 0.5:
            pitched_up = pitch_shift(base_synth, 220.0, base_freq*2.0, measure_dur)
            pitched += pitched_up * 0.5

        # user-defined volume for the synth
        synth_vol = global_params["synth_volume"]
        out[:len(pitched)] += pitched[:len(out)] * synth_vol

        return out


# =============================================================================
# CHORD / HARMONY
# =============================================================================

DIM_OR_AUG_RARE_WEIGHT = 0.05

CHORD_INTERVALS = {
    "maj":  ([0, 4, 7], 1.0),
    "min":  ([0, 3, 7], 1.0),
    "7":    ([0, 4, 7, 10], 0.9),
    "maj7": ([0, 4, 7, 11], 0.9),
    "min7": ([0, 3, 7, 10], 0.9),
    "dim":  ([0, 3, 6], DIM_OR_AUG_RARE_WEIGHT),
    "aug":  ([0, 4, 8], DIM_OR_AUG_RARE_WEIGHT),
    "9":    ([0, 4, 7, 10, 14], 0.4),
    "11":   ([0, 4, 7, 10, 14, 17], 0.3),
    "13":   ([0, 4, 7, 10, 14, 17, 21], 0.3),
    "sus2": ([0, 2, 7], 0.5),
    "sus4": ([0, 5, 7], 0.5),
}

ALL_MODES = ["major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian"]

MODE_SCALE_OFFSETS = {
    "major":       [0, 2, 4, 5, 7, 9, 11],
    "minor":       [0, 2, 3, 5, 7, 8, 10],
    "dorian":      [0, 2, 3, 5, 7, 9, 10],
    "phrygian":    [0, 1, 3, 5, 7, 8, 10],
    "lydian":      [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":  [0, 2, 4, 5, 7, 9, 10],
    "aeolian":     [0, 2, 3, 5, 7, 8, 10],  # natural minor
    "locrian":     [0, 1, 3, 5, 6, 8, 10]
}

def get_scale_midi(root, mode_name):
    offsets = MODE_SCALE_OFFSETS.get(mode_name, MODE_SCALE_OFFSETS["major"])
    return [root + x for x in offsets]

def chord_name_to_frequencies(root_midi, chord_type):
    intervals, base_weight = CHORD_INTERVALS[chord_type]
    chord_freqs = []
    base_root = root_midi - 12
    for st in intervals:
        note_midi = base_root + st
        freq = midi_note_to_freq(note_midi)
        chord_freqs.append(freq)

    ext_val = global_params["chord_extensions"]
    if ext_val > 0.5:
        if random.random() < ext_val:
            chord_freqs.append(midi_note_to_freq(base_root - 12))
        if random.random() < ext_val and len(intervals) > 1:
            high_midi = base_root + intervals[1] + 12
            chord_freqs.append(midi_note_to_freq(high_midi))

    chord_freqs.sort()
    return chord_freqs

def pick_chord_type():
    d_val = global_params["chord_dissonance"]
    all_types = list(CHORD_INTERVALS.keys())
    while True:
        ctype = random.choice(all_types)
        if ctype in ["dim", "aug"]:
            if random.random() < d_val:
                return ctype
            else:
                continue
        else:
            # skip 9/11/13 if chord_extensions < 0.5
            if ctype in ["9", "11", "13"] and global_params["chord_extensions"] < 0.5:
                continue
            return ctype


# =============================================================================
# DRUM PATTERNS
# =============================================================================

def generate_drums(m_len, measure_index):
    out = np.zeros(m_len, dtype=np.float32)
    busyness = global_params["drum_busyness"]

    place_drum_hit(out, 'kick', 0.0)
    if random.random() < busyness:
        place_drum_hit(out, 'kick', 2.0)
    place_drum_hit(out, 'snare', 1.98)
    place_drum_hit(out, 'snare', 3.98)

    # hats
    if busyness > 0.5:
        for i in range(BEATS_PER_MEASURE*2):
            hat_time = i*0.5
            if random.random() < 0.85:
                place_drum_hit(out, 'hat', hat_time)
    else:
        for i in range(BEATS_PER_MEASURE):
            hat_time = float(i)
            if random.random() < 0.8:
                place_drum_hit(out, 'hat', hat_time)

    if (measure_index % 8) == 0 and random.random() < (0.2 + busyness*0.5):
        place_drum_hit(out, 'snare', 3.5)
        place_drum_hit(out, 'snare', 3.75)

    out *= global_params["drum_volume"]
    return out


# =============================================================================
# BASS
# =============================================================================

def generate_simple_bass(freq, dur=0.4):
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, endpoint=False)
    wave = np.sin(2*np.pi*freq*t)
    env = np.exp(-3 * t)
    bass_wave = (wave * env * 0.4).astype(np.float32)
    bass_wave *= global_params["bass_volume"]
    return bass_wave

def generate_bass_line(chord_freqs, m_len):
    out = np.zeros(m_len, dtype=np.float32)
    if not chord_freqs:
        return out
    root_freq = chord_freqs[0]
    alt_freq = chord_freqs[1] if len(chord_freqs) > 1 else root_freq

    for beat in [0.0, 2.0]:
        freq = root_freq if beat == 0.0 else alt_freq
        wave = generate_simple_bass(freq, 0.4)
        start_idx = beats_to_samples(beat)
        end_idx = min(start_idx + len(wave), m_len)
        out[start_idx:end_idx] += wave[:end_idx - start_idx]
    return out


# =============================================================================
# CHORD RHYTHM
# =============================================================================

def generate_chord_rhythm_line(chord_freqs, m_len):
    out = np.zeros(m_len, dtype=np.float32)
    if not chord_freqs:
        return out
    hits = [(0.0, 1.0), (2.0, 1.0)]
    for (start_beat, dur_beats) in hits:
        chord_dur_sec = dur_beats * (60.0 / get_bpm())
        chord_wave = np.zeros(int(chord_dur_sec * SAMPLE_RATE), dtype=np.float32)
        for f in chord_freqs:
            partial = generate_piano_note(f, chord_dur_sec)
            chord_wave[:len(partial)] += partial[:len(chord_wave)]
        a_len = int(0.02 * len(chord_wave))
        d_len = int(0.1 * len(chord_wave))
        for i in range(a_len):
            chord_wave[i] *= (i / a_len)
        for i in range(d_len):
            chord_wave[-1 - i] *= (i / d_len)

        start_idx = beats_to_samples(start_beat)
        end_idx = min(start_idx + len(chord_wave), m_len)
        out[start_idx:end_idx] += chord_wave[:end_idx - start_idx]

    return out


# =============================================================================
# MELODY
# =============================================================================

def generate_melody_line(chord_freqs, m_len):
    out = np.zeros(m_len, dtype=np.float32)
    if not chord_freqs:
        return out
    times = [0.0]
    if random.random() < 0.5:
        times.append(2.0)
    for beat in times:
        freq = random.choice(chord_freqs)
        note_dur = 1.0
        note_wave = generate_piano_note(freq, note_dur*(60.0 / get_bpm()))
        start_idx = beats_to_samples(beat)
        end_idx = min(start_idx + len(note_wave), m_len)
        out[start_idx:end_idx] += note_wave[:end_idx - start_idx]
    return out


# =============================================================================
# SECTION LENGTH
# =============================================================================

MIN_SECTION_LENGTH = 10
MAX_SECTION_LENGTH = 14


# =============================================================================
# CHORD PROGRESSION MANAGER
# =============================================================================

class ChordProgressionManager:
    def __init__(self):
        self.measure_count = 0
        self.chord_history = []
        self.current_chords = []
        self.current_chord_index = 0
        self.current_section_length = random.randint(MIN_SECTION_LENGTH, MAX_SECTION_LENGTH)
        self.measure_in_section = 0

        # Reintroduce a background synth manager:
        self.synth_mgr = BackgroundSynthManager()

        self.generate_new_section()

    def generate_new_section(self):
        # We'll check if random change to mode if mode_cycle_speed is not small
        cycle_speed = global_params["mode_cycle_speed"]
        # e.g. if cycle_speed=0.05 => 5% chance or even smaller
        if random.random() < cycle_speed:
            new_mode = random.choice(ALL_MODES)
            global_params["current_mode"] = new_mode
            print(f"Auto-switching mode to: {new_mode}")

        self.current_chords = []
        length = random.choice([4, 4, 5])
        scale_midi = get_scale_midi(global_params["key_root_midi"], global_params["current_mode"])
        for _ in range(length):
            ctype = pick_chord_type()
            chord_root = random.choice(scale_midi)
            cdict = {
                "midi_root": chord_root,
                "chord_type": ctype
            }
            self.current_chords.append(cdict)

        self.current_section_length = random.randint(MIN_SECTION_LENGTH, MAX_SECTION_LENGTH)
        self.measure_in_section = 0
        self.current_chord_index = 0

        print("=== NEW SECTION ===")
        for c in self.current_chords:
            print(" ->", c)

    def get_next_chord(self):
        if self.measure_in_section >= self.current_section_length:
            self.generate_new_section()

        chord_dict = self.current_chords[self.current_chord_index]
        chord_freqs = chord_name_to_frequencies(chord_dict["midi_root"], chord_dict["chord_type"])

        self.chord_history.append(chord_dict)
        self.current_chord_index = (self.current_chord_index + 1) % len(self.current_chords)
        self.measure_in_section += 1
        self.measure_count += 1

        return chord_freqs, chord_dict, self.synth_mgr.generate_pad_for_chord(chord_freqs)


# =============================================================================
# MEASURE GENERATION
# =============================================================================

def generate_one_measure(chord_mgr):
    m_len = measure_len_samples()

    chord_freqs, chord_dict, synth_wave = chord_mgr.get_next_chord()

    chord_wave = generate_chord_rhythm_line(chord_freqs, m_len)
    drum_wave = generate_drums(m_len, chord_mgr.measure_count)
    bass_wave = generate_bass_line(chord_freqs, m_len)
    melody_wave = generate_melody_line(chord_freqs, m_len)

    combined = chord_wave + drum_wave + bass_wave + melody_wave + synth_wave

    filtered = low_pass_filter(combined)
    noise = add_vinyl_noise(len(filtered))
    result = filtered + noise

    master_vol = get_master_volume()
    result *= master_vol

    peak = np.max(np.abs(result))
    if peak > 1.0:
        result /= peak
    result *= db_to_linear(DECIBEL_HEADROOM)

    return result.astype(np.float32)


# =============================================================================
# BACKGROUND PRODUCER THREAD
# =============================================================================

audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

def producer_thread(chord_mgr):
    while True:
        measure_audio = generate_one_measure(chord_mgr)
        idx = 0
        while idx < len(measure_audio):
            block = measure_audio[idx:idx+CHUNK_SIZE]
            audio_queue.put(block, block=True)
            idx += CHUNK_SIZE


# =============================================================================
# AUDIO CALLBACK
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
# TKINTER UI
# =============================================================================

class LofiUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lo-fi Controls with Sweeping Synth")

        # Create a ScrollableFrame
        self.scrollable_frame = ScrollableFrame(self.root)
        self.scrollable_frame.pack(fill="both", expand=True)

        # Inside the interior frame, we add our controls:
        f = self.scrollable_frame.interior

        self.add_slider(f, "Master Volume", "master_volume", 0, 1, 0.5)
        self.add_slider(f, "Piano Volume", "piano_volume", 0, 1, 0.5)
        self.add_slider(f, "Bass Volume", "bass_volume", 0, 1, 0.5)
        self.add_slider(f, "Drum Volume", "drum_volume", 0, 1, 0.5)
        self.add_slider(f, "Vinyl Noise", "noise_level", 0, 1, 0.5)
        self.add_slider(f, "Tempo (BPM)", "tempo", 40, 160, 70)
        self.add_slider(f, "Dissonance", "chord_dissonance", 0, 1, 0.5)
        self.add_slider(f, "Chord Extensions", "chord_extensions", 0, 1, 0.5)
        self.add_slider(f, "Drum Busyness", "drum_busyness", 0, 1, 0.5)
        # Mode cycle speed is small by default => rare auto-mode changes
        self.add_slider(f, "Mode Cycle Speed", "mode_cycle_speed", 0, 1, 0.05)
        # Sweeping synth volume
        self.add_slider(f, "Synth Volume", "synth_volume", 0, 1, 0.4)

        # Current mode
        mode_label = tk.Label(f, text="Current Mode")
        mode_label.pack()
        self.mode_var = tk.StringVar(value=global_params["current_mode"])
        mode_menu = tk.OptionMenu(f, self.mode_var, *ALL_MODES, command=self.update_mode)
        mode_menu.pack()

        # Key root
        self.add_slider(f, "Key Root MIDI", "key_root_midi", 36, 72, 48)

        # We can place more widgets if needed
        self.root.after(100, self.update_loop)

    def add_slider(self, parent, label_text, param_name, from_, to_, default):
        label = tk.Label(parent, text=label_text)
        label.pack()
        resolution = 0.01
        # If the range is large & integral, you might want resolution=1
        scale = tk.Scale(parent, from_=from_, to=to_,
                         orient=tk.HORIZONTAL, resolution=resolution,
                         command=lambda val, p=param_name: self.on_slider_change(p, val))
        scale.set(default)
        scale.pack()

    def on_slider_change(self, param_name, value):
        global_params[param_name] = float(value)

    def update_mode(self, new_mode):
        global_params["current_mode"] = new_mode
        print(f"User set mode to: {new_mode}")

    def update_loop(self):
        # Could do real-time UI updates, but for now we just poll every 100ms
        self.root.after(100, self.update_loop)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Lo-fi jam starting...")
    chord_mgr = ChordProgressionManager()

    # Producer thread
    thread = threading.Thread(
        target=producer_thread,
        args=(chord_mgr,),
        daemon=True
    )
    thread.start()

    # Audio stream
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        channels=1,
        dtype='float32',
        callback=audio_callback
    ):
        # Launch tkinter UI
        root = tk.Tk()
        ui = LofiUI(root)
        root.mainloop()

if __name__ == '__main__':
    main()
