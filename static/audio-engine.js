/**
 * Cyberpunk Audio Engine v2 — Web Audio Synthesis
 * v1.8.0 — 2026-04-11
 *
 * Why rewritten:
 *   The v1 engine fired a random square-wave note on every 16th step
 *   (10 notes/sec at 150 BPM) producing a clangy, atonal noise wall.
 *   This v2 replaces that with a structured, atmospheric soundscape that
 *   still fits the 4-5 second auto-play window:
 *
 *   OLD → NEW
 *   ─────────────────────────────────────────────────────────────────
 *   Random square arpeggios every 16th  → Sparse triangle pluck melody
 *                                          (8-step pattern, mostly rests)
 *   Single random beep every 2 beats    → Chord pad wash (E+B, then A+D)
 *   2-oscillator E1 drone               → 3-oscillator drone (root+detune+fifth)
 *   No percussion                       → Soft sine kick + whisper hi-hat
 *   ─────────────────────────────────────────────────────────────────
 *
 * Public API is unchanged: toggle(), playSwoosh(), setIntensity(),
 * glitchWin(), pitchSlideWin()
 */
const AudioEngine = {
    ctx: null,
    isPlaying: false,
    bpm: 128,
    nextNoteTime: 0,
    scheduleAheadTime: 0.1,
    lookahead: 25,
    timerID: null,
    current16thNote: 0,

    prophet: null,
    prophetFilter: null,
    noiseBuffer: null,

    // E Minor Pentatonic: E3, G3, A3, B3, D4
    scale: [164.81, 196.00, 220.00, 246.94, 293.66],

    // 8 melody slots (one per pair of 16th notes); -1 = rest
    // Reads as: E3 … A3 … D4, B3 … G3 …
    melodyPattern: [0, -1, 2, -1, 4, 3, -1, 1],

    // ── Initialisation ──────────────────────────────────────────────────

    init() {
        if (this.ctx) return;
        this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        this.setupProphet();

        // Pre-bake white-noise buffer (shared by swoosh + hihat)
        const bufferSize = 2 * this.ctx.sampleRate;
        this.noiseBuffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
        const data = this.noiseBuffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) data[i] = Math.random() * 2 - 1;
    },

    // ── Drone ────────────────────────────────────────────────────────────

    setupProphet() {
        this.prophetFilter = this.ctx.createBiquadFilter();
        this.prophetFilter.type = 'lowpass';
        this.prophetFilter.frequency.value = 480;
        this.prophetFilter.Q.value = 7;
        this.prophetFilter.connect(this.ctx.destination);
    },

    playProphet() {
        const osc1 = this.ctx.createOscillator(); // E1 sawtooth — gritty sub
        const osc2 = this.ctx.createOscillator(); // E1 square   — slightly detuned
        const osc3 = this.ctx.createOscillator(); // B1 triangle — fifth for warmth
        const gain = this.ctx.createGain();

        osc1.type = 'sawtooth'; osc1.frequency.value = 41.20;
        osc2.type = 'square';   osc2.frequency.value = 41.55; // 35-cent detune
        osc3.type = 'triangle'; osc3.frequency.value = 61.74; // B1

        gain.gain.setValueAtTime(0, this.ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0.20, this.ctx.currentTime + 2.0); // slow bloom

        [osc1, osc2, osc3].forEach(o => { o.connect(gain); o.start(); });
        gain.connect(this.prophetFilter);
        this.prophet = { osc1, osc2, osc3, gain };
    },

    stopProphet() {
        if (!this.prophet) return;
        const now = this.ctx.currentTime;
        this.prophet.gain.gain.cancelScheduledValues(now);
        this.prophet.gain.gain.linearRampToValueAtTime(0, now + 0.8);
        setTimeout(() => {
            if (this.prophet) {
                this.prophet.osc1.stop();
                this.prophet.osc2.stop();
                this.prophet.osc3.stop();
                this.prophet = null;
            }
        }, 900);
    },

    // ── Move sound (on every piece drop) ────────────────────────────────
    // row: 0 = top of board (light fall), 5 = bottom (heavy thud).
    // Higher rows → lighter whoosh (higher pitch, shorter decay).
    // Lower rows → deeper whoosh + heavier sub-thud, longer decay.
    // ±10 % pitch jitter makes every drop feel slightly unique.

    playSwoosh(row) {
        if (!this.ctx) this.init();
        if (this.ctx.state === 'suspended') this.ctx.resume();

        // row 0 (top) → t=0.0, row 5 (bottom) → t=1.0
        const t = Math.min(Math.max((row || 0) / 5, 0), 1);
        // ±10 % pitch randomness
        const jitter = 0.9 + Math.random() * 0.2;

        // Whoosh: frequency sweeps from high to low; bottom rows start lower
        const freqStart = jitter * (1100 - t * 500);   // 1100 → 600 Hz range start
        const freqEnd   = jitter * (220  - t * 130);   //  220 →  90 Hz range end
        const duration  = 0.18 + t * 0.12;             // 0.18 → 0.30 s

        const noise  = this.ctx.createBufferSource();
        const filter = this.ctx.createBiquadFilter();
        const gain   = this.ctx.createGain();

        noise.buffer   = this.noiseBuffer;
        filter.type    = 'bandpass';
        filter.Q.value = 1.5 + t * 0.8;               // warmer Q on lower rows
        filter.frequency.setValueAtTime(freqStart, this.ctx.currentTime);
        filter.frequency.exponentialRampToValueAtTime(freqEnd, this.ctx.currentTime + duration);

        gain.gain.setValueAtTime(0, this.ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0.13 + t * 0.07, this.ctx.currentTime + 0.008);
        gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + duration + 0.05);

        noise.connect(filter);
        filter.connect(gain);
        gain.connect(this.ctx.destination);
        noise.start();
        noise.stop(this.ctx.currentTime + duration + 0.08);

        // Sub-thud: deeper and louder on lower rows, almost silent on top rows
        if (t > 0.05) {
            const thudFreq = jitter * (180 - t * 120); // 180 → 60 Hz
            const thudGain = 0.08 + t * 0.38;          // 0.08 → 0.46
            const thudLen  = 0.25 + t * 0.18;          // 0.25 → 0.43 s

            const osc  = this.ctx.createOscillator();
            const tGain = this.ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(thudFreq, this.ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(thudFreq * 0.3, this.ctx.currentTime + thudLen);
            tGain.gain.setValueAtTime(thudGain, this.ctx.currentTime);
            tGain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + thudLen);
            osc.connect(tGain);
            tGain.connect(this.ctx.destination);
            osc.start(this.ctx.currentTime);
            osc.stop(this.ctx.currentTime + thudLen + 0.05);
        }
    },

    // ── Background rhythm voices ─────────────────────────────────────────

    playKick(time) {
        const osc  = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(110, time);
        osc.frequency.exponentialRampToValueAtTime(0.01, time + 0.3);
        gain.gain.setValueAtTime(0.5, time);
        gain.gain.exponentialRampToValueAtTime(0.0001, time + 0.35);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start(time);
        osc.stop(time + 0.38);
    },

    playHihat(time) {
        const noise  = this.ctx.createBufferSource();
        const filter = this.ctx.createBiquadFilter();
        const gain   = this.ctx.createGain();
        noise.buffer     = this.noiseBuffer;
        filter.type      = 'highpass';
        filter.frequency.value = 9000;
        gain.gain.setValueAtTime(0.035, time);
        gain.gain.exponentialRampToValueAtTime(0.0001, time + 0.03);
        noise.connect(filter);
        filter.connect(gain);
        gain.connect(this.ctx.destination);
        noise.start(time);
        noise.stop(time + 0.04);
    },

    // Atmospheric chord pad — three detuned sines, slow bloom
    playPad(time, freq) {
        const master = this.ctx.createGain();
        master.gain.setValueAtTime(0, time);
        master.gain.linearRampToValueAtTime(0.05, time + 0.4);
        master.gain.exponentialRampToValueAtTime(0.0001, time + 1.5);
        master.connect(this.ctx.destination);

        [-12, 0, 12].forEach(cents => {
            const osc = this.ctx.createOscillator();
            osc.type = 'sine';
            osc.frequency.value = freq;
            osc.detune.value = cents;
            osc.connect(master);
            osc.start(time);
            osc.stop(time + 1.6);
        });
    },

    // Pluck melody — triangle with quick attack and warm filter sweep
    playPluck(time, freq) {
        const osc    = this.ctx.createOscillator();
        const filter = this.ctx.createBiquadFilter();
        const gain   = this.ctx.createGain();

        osc.type = 'triangle';
        osc.frequency.value = freq;

        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(3200, time);
        filter.frequency.exponentialRampToValueAtTime(500, time + 0.4);

        gain.gain.setValueAtTime(0, time);
        gain.gain.linearRampToValueAtTime(0.12, time + 0.006);
        gain.gain.exponentialRampToValueAtTime(0.0001, time + 0.5);

        osc.connect(filter);
        filter.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start(time);
        osc.stop(time + 0.55);
    },

    // Retained for win-effect use only (called externally)
    playBeep(time) {
        const osc  = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(880,  time);
        osc.frequency.exponentialRampToValueAtTime(1760, time + 0.08);
        gain.gain.setValueAtTime(0, time);
        gain.gain.linearRampToValueAtTime(0.05, time + 0.01);
        gain.gain.exponentialRampToValueAtTime(0.0001, time + 0.25);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start(time);
        osc.stop(time + 0.28);
    },

    // ── Scheduler ────────────────────────────────────────────────────────

    nextNote() {
        const secondsPerBeat = 60.0 / this.bpm;
        this.nextNoteTime += 0.25 * secondsPerBeat; // advance one 16th note
        this.current16thNote = (this.current16thNote + 1) % 16;
    },

    scheduleNote(beatNumber, time) {
        const beat = beatNumber % 16;

        // Kick on beat 0 (bar start) and beat 8 (half-bar)
        if (beat === 0 || beat === 8) this.playKick(time);

        // Whisper hi-hat on all off-beats
        if (beat % 2 === 1) this.playHihat(time);

        // Chord pad: E3+B3 on beat 0, A3+D4 on beat 8 (harmonic shift)
        if (beat === 0) {
            this.playPad(time, this.scale[0]); // E3
            this.playPad(time, this.scale[3]); // B3
        }
        if (beat === 8) {
            this.playPad(time, this.scale[2]); // A3
            this.playPad(time, this.scale[4]); // D4
        }

        // Sparse pluck: one trigger every two 16th notes (8 slots per bar)
        if (beat % 2 === 0) {
            const slot    = beat / 2;               // 0..7
            const noteIdx = this.melodyPattern[slot];
            if (noteIdx >= 0) this.playPluck(time, this.scale[noteIdx]);
        }
    },

    scheduler() {
        while (this.nextNoteTime < this.ctx.currentTime + this.scheduleAheadTime) {
            this.scheduleNote(this.current16thNote, this.nextNoteTime);
            this.nextNote();
        }
        this.timerID = setTimeout(() => this.scheduler(), this.lookahead);
    },

    // ── Public API ────────────────────────────────────────────────────────

    toggle() {
        this.init();
        if (this.isPlaying) {
            this.isPlaying = false;
            clearTimeout(this.timerID);
            this.stopProphet();
        } else {
            this.isPlaying = true;
            this.ctx.resume();
            this.current16thNote = 0;
            this.nextNoteTime = this.ctx.currentTime;
            this.playProphet();
            this.scheduler();
        }
        return this.isPlaying;
    },

    // Open the drone filter for high-scoring moves
    setIntensity(score) {
        if (!this.ctx || !this.prophetFilter) return;
        const freq = 350 + score * 350;
        this.prophetFilter.frequency.setTargetAtTime(freq, this.ctx.currentTime, 0.5);
    },

    // ── Menace sting ─────────────────────────────────────────────────────
    // Low tritone stab (E2 + A#2 sawtooth) + deep tremolo rumble.
    // Fires at most once per 4 seconds to avoid spam.
    _lastMenace: 0,

    playMenace() {
        if (!this.ctx) this.init();
        if (this.ctx.state === 'suspended') this.ctx.resume();

        const now = this.ctx.currentTime;
        // 4-second cooldown
        if (now - this._lastMenace < 4) return;
        this._lastMenace = now;

        // Tritone stab: E2 (82.4 Hz) + A#2 (116.5 Hz), sawtooth
        [82.4, 116.5].forEach(freq => {
            const osc  = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.value = freq;
            gain.gain.setValueAtTime(0, now);
            gain.gain.linearRampToValueAtTime(0.18, now + 0.04);
            gain.gain.exponentialRampToValueAtTime(0.0001, now + 1.2);
            osc.connect(gain);
            gain.connect(this.prophetFilter);
            osc.start(now);
            osc.stop(now + 1.3);
        });

        // Deep tremolo rumble: A1 (55 Hz) amplitude-modulated at 8 Hz
        const rumble = this.ctx.createOscillator();
        const lfo    = this.ctx.createOscillator();
        const lfoAmp = this.ctx.createGain();
        const env    = this.ctx.createGain();

        rumble.type = 'sine';
        rumble.frequency.value = 55;

        lfo.type = 'sine';
        lfo.frequency.value = 8;           // 8 Hz tremolo
        lfoAmp.gain.value = 0.08;          // modulation depth

        env.gain.setValueAtTime(0, now);
        env.gain.linearRampToValueAtTime(0.14, now + 0.08);
        env.gain.exponentialRampToValueAtTime(0.0001, now + 1.5);

        lfo.connect(lfoAmp);
        lfoAmp.connect(env.gain);          // LFO modulates envelope
        rumble.connect(env);
        env.connect(this.ctx.destination);

        lfo.start(now);    lfo.stop(now + 1.6);
        rumble.start(now); rumble.stop(now + 1.6);
    },

    // Win effects (called from win-effects.js)
    glitchWin() {
        if (!this.ctx || !this.isPlaying) return;
        const orig = this.bpm;
        this.bpm = 256;
        setTimeout(() => { this.bpm = 64; setTimeout(() => { this.bpm = orig; }, 500); }, 100);
    },

    pitchSlideWin() {
        if (!this.ctx || !this.prophet) return;
        const now = this.ctx.currentTime;
        ['osc1', 'osc2', 'osc3'].forEach(k => {
            if (this.prophet[k])
                this.prophet[k].frequency.exponentialRampToValueAtTime(10, now + 1.5);
        });
        this.prophet.gain.gain.exponentialRampToValueAtTime(0.0001, now + 1.5);
    }
};
