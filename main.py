import numpy as np
import random
import math
import queue
import threading
import time

import sounddevice as sd
from scipy.signal import butter, filtfilt

# =============================================================================
# Config
# =============================================================================
SAMPLE_RATE = 44100
BPM = 70
BEATS_PER_MEASURE = 4
MEASURE_SECONDS = BEATS_PER_MEASURE * (60.0 / BPM)

# We'll break each measure into smaller blocks for the queue.
# The callback reads from the queue chunk by chunk.
# If we do a big chunk (e.g., entire measure) that also works, but let's
# keep it smaller in case we want more frequent refills.
CHUNK_SIZE = 2048   # ~46ms per chunk @ 44100 Hz

# We keep a queue that can hold a few measures worth of chunks so we don't starve.
MAX_QUEUE_SIZE = 16  # in chunks

# A little final volume headroom in dB
DECIBEL_HEADROOM = -3.0


# =============================================================================
# Utility Functions
# =============================================================================

def db_to_linear(db):
    """Convert decibels to a linear amplitude factor."""
    return 10 ** (db / 20.0)

def low_pass_filter(signal, cutoff_freq, sample_rate=SAMPLE_RATE):
    """
    A simple 2nd-order Butterworth low-pass filter.
    """
    b, a = butter(N=2, Wn=cutoff_freq/(0.5 * sample_rate), btype='low')
    return filtfilt(b, a, signal)

def add_vinyl_noise(num_samples, noise_level=0.02):
    """
    Generate random 'vinyl crackle' style noise.
    """
    noise = np.zeros(num_samples, dtype=np.float32)
    # ~2% random clicks
    mask = np.random.choice([0, 1], size=num_samples, p=[0.98, 0.02])
    # Each click has random amplitude in [-1..1]
    vals = (np.random.rand(num_samples) - 0.5) * 2
    noise = vals * mask * noise_level
    return noise


# =============================================================================
# Instrument Generators
# =============================================================================

def generate_kick(dur=0.4):
    """
    Simple kick drum: sine wave sweep from ~100Hz to ~40Hz, exponential decay.
    """
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, endpoint=False)
    freq_sweep = np.linspace(100, 40, n)
    wave = np.sin(2*np.pi * freq_sweep * t)
    env = np.exp(-5 * t)
    return wave * env

def generate_snare(dur=0.2):
    """
    Simple snare: white noise + short sine 'body'.
    """
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, endpoint=False)
    noise = np.random.randn(n) * 0.5
    body = 0.5 * np.sin(2*np.pi * 200 * t)
    env = np.exp(-10 * t)
    return (noise + body) * env

def generate_hat(dur=0.1):
    """
    Simple hi-hat: filtered noise with very quick decay.
    """
    n = int(dur * SAMPLE_RATE)
    noise = np.random.randn(n)
    # do a simple 'band-pass' style by subtracting a short average
    filtered = noise - np.convolve(noise, np.ones(5)/5, mode='same')
    env = np.exp(-20 * np.linspace(0, dur, n))
    return filtered * env * 0.3

def generate_chord(freqs, dur=1.0, amplitude=0.3):
    """
    Simple chord from summing sines at given freqs (minor triad).
    """
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, endpoint=False)
    result = np.zeros(n, dtype=np.float32)
    for f in freqs:
        result += (amplitude * np.sin(2*np.pi * f * t)).astype(np.float32)
    
    # simple fade in/out
    attack_len = int(0.1 * n)
    decay_len = int(0.2 * n)
    for i in range(attack_len):
        result[i] *= (i / attack_len)
    for i in range(decay_len):
        result[-1 - i] *= (i / decay_len)

    return result


# =============================================================================
# Beat / Pattern for ONE measure
# =============================================================================

def generate_random_chord():
    """Pick a random minor triad from A-minor scale."""
    # A minor scale ~ A(220), B(247), C(261.6), D(293.7), E(329.6), F(349.2), G(392)
    base_freqs = [220.0, 246.94, 261.63, 293.66, 329.63, 349.23, 392.0]
    root = random.choice(base_freqs)
    major_or_minor = random.choice([0, 1])
    if major_or_minor == 0:
        third = root * 1.25
    else:
        third = root * 1.2
    perfect_fifth = root * 1.50
    minor_seventh = root * 1.75
    return [root, third, perfect_fifth, minor_seventh]

def generate_beat_events():
    """
    Return a list of (time_in_beats, 'kick'/'snare'/'hat').
    4/4 measure, random variations.
    """
    events = []
    # Kick on beat 1
    events.append((0.0, 'kick'))
    # Maybe an extra kick somewhere
    if random.random() < 0.5:
        events.append((random.choice([1.5, 2, 2.5, 3]), 'kick'))
    # Snare near beats 2 & 4
    events.append((1.98, 'snare'))
    events.append((3.98, 'snare'))
    # Hats on 8th notes
    for i in range(BEATS_PER_MEASURE * 2):
        hat_time = i * 0.5
        if random.random() < 0.85:
            events.append((hat_time, 'hat'))
    events.sort(key=lambda x: x[0])
    return events

def generate_one_measure():
    """
    Generate a float32 array (one measure of audio).
    """
    measure_len = int(MEASURE_SECONDS * SAMPLE_RATE)
    out = np.zeros(measure_len, dtype=np.float32)

    # 1) chord
    chord_freqs = generate_random_chord()
    chord_wave = generate_chord(chord_freqs, dur=MEASURE_SECONDS, amplitude=0.3)
    out += chord_wave[:measure_len]

    # 2) drums
    events = generate_beat_events()
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
            continue

        end_idx = min(start_idx + len(wave), measure_len)
        wave_len = end_idx - start_idx
        out[start_idx:end_idx] += wave[:wave_len].astype(np.float32)

    # 3) low-pass
    filtered = low_pass_filter(out, cutoff_freq=5000)

    # 4) vinyl noise
    crackle = add_vinyl_noise(len(filtered), noise_level=0.02)
    result = filtered + crackle

    # 5) normalize if necessary, then apply headroom
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result /= peak
    result *= db_to_linear(DECIBEL_HEADROOM)

    return result.astype(np.float32)


# =============================================================================
# Background Producer Thread
# =============================================================================

audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

def producer_thread():
    """
    Continuously generate measures, slice them into chunks, and put them into the queue.
    """
    while True:
        measure = generate_one_measure()
        # slice the measure into CHUNK_SIZE blocks
        idx = 0
        while idx < len(measure):
            block = measure[idx:idx+CHUNK_SIZE]
            audio_queue.put(block, block=True)  # blocks if the queue is full
            idx += CHUNK_SIZE
        # then loop around and generate another measure


# =============================================================================
# Callback
# =============================================================================

def audio_callback(outdata, frames, time_info, status):
    if status:
        print("Status:", status)
    try:
        # We expect 'frames' == CHUNK_SIZE, but let's handle partial just in case
        block = audio_queue.get(block=True)  # blocks if the queue is empty
        # If block is smaller than frames, pad it
        if len(block) < frames:
            out = np.zeros(frames, dtype=np.float32)
            out[:len(block)] = block
        else:
            out = block[:frames]

        # If we had stereo, we'd do outdata[:,0] = out, outdata[:,1] = out for dual mono
        outdata[:, 0] = out
    except Exception as e:
        print("Exception in callback:", e)
        # fill outdata with zeros to avoid horrible noise
        outdata.fill(0)


# =============================================================================
# Main
# =============================================================================

def main():
    print("Starting infinite lofi jam with background generation...")

    # Start the producer thread
    thread = threading.Thread(target=producer_thread, daemon=True)
    thread.start()

    # Start the audio stream
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,   # the frames passed to callback
        channels=1,            # mono
        dtype='float32',
        callback=audio_callback
    ):
        # Just keep the main thread alive
        try:
            while True:
                time.sleep(1)  # Sleep main thread; audio runs in callback
        except KeyboardInterrupt:
            print("Exiting on Ctrl+C...")


if __name__ == '__main__':
    main()
