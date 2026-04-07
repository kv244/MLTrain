/**
 * Manga Sci-Fi Audio Engine: Real-time Web Audio Synthesis
 * Mimics Sonic Pi script @ 150 BPM
 */
const AudioEngine = {
    ctx: null,
    isPlaying: false,
    bpm: 150,
    startTime: 0,
    nextNoteTime: 0,
    scheduleAheadTime: 0.1,
    lookahead: 25,
    timerID: null,
    current16thNote: 0,
    
    // Synths
    prophet: null,
    prophetFilter: null,
    noiseBuffer: null,
    
    // Scale: E Minor Pentatonic (E, G, A, B, D)
    scale: [164.81, 196.00, 220.00, 246.94, 293.66], // E3, G3, A3, B3, D4
    
    init() {
        if (this.ctx) return;
        this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        this.setupProphet();

        // ── Pre-generate white noise for the "swoosh" ─────────────────────
        const bufferSize = 2 * this.ctx.sampleRate;
        this.noiseBuffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
        const output = this.noiseBuffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) {
            output[i] = Math.random() * 2 - 1;
        }
    },

    playSwoosh() {
        if (!this.ctx) this.init();
        if (this.ctx.state === 'suspended') this.ctx.resume();

        const noise = this.ctx.createBufferSource();
        const filter = this.ctx.createBiquadFilter();
        const gain = this.ctx.createGain();

        noise.buffer = this.noiseBuffer;

        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(3000, this.ctx.currentTime);
        filter.frequency.exponentialRampToValueAtTime(200, this.ctx.currentTime + 0.2);

        gain.gain.setValueAtTime(0, this.ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0.3, this.ctx.currentTime + 0.01);
        gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + 0.25);

        noise.connect(filter);
        filter.connect(gain);
        gain.connect(this.ctx.destination);

        noise.start();
        noise.stop(this.ctx.currentTime + 0.25);
    },

    setupProphet() {
        // Create a deep "Robotic Core" drone (E1 approx 41.20Hz)
        this.prophetFilter = this.ctx.createBiquadFilter();
        this.prophetFilter.type = 'lowpass';
        this.prophetFilter.frequency.value = 400;
        this.prophetFilter.Q.value = 10;
        this.prophetFilter.connect(this.ctx.destination);
    },

    playProphet() {
        const osc1 = this.ctx.createOscillator();
        const osc2 = this.ctx.createOscillator();
        const gain = this.ctx.createGain();

        osc1.type = 'sawtooth';
        osc2.type = 'square';
        
        // E1 Drone (41.20 Hz)
        osc1.frequency.value = 41.20;
        osc2.frequency.value = 41.50; // Sligh detune for prophet-feel

        gain.gain.setValueAtTime(0, this.ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0.3, this.ctx.currentTime + 1);
        
        osc1.connect(gain);
        osc2.connect(gain);
        gain.connect(this.prophetFilter);
        
        osc1.start();
        osc2.start();
        this.prophet = { osc1, osc2, gain };
    },

    stopProphet() {
        if (!this.prophet) return;
        const now = this.ctx.currentTime;
        this.prophet.gain.gain.cancelScheduledValues(now);
        this.prophet.gain.gain.linearRampToValueAtTime(0, now + 0.5);
        setTimeout(() => {
            if (this.prophet) {
                this.prophet.osc1.stop();
                this.prophet.osc2.stop();
                this.prophet = null;
            }
        }, 600);
    },

    playChip(time, freq) {
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        const filter = this.ctx.createBiquadFilter();

        osc.type = 'square';
        osc.frequency.setValueAtTime(freq, time);
        
        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(2000, time);
        filter.frequency.exponentialRampToValueAtTime(100, time + 0.1);

        gain.gain.setValueAtTime(0, time);
        gain.gain.linearRampToValueAtTime(0.15, time + 0.01);
        gain.gain.exponentialRampToValueAtTime(0.0001, time + 0.1);

        osc.connect(filter);
        filter.connect(gain);
        gain.connect(this.ctx.destination);

        osc.start(time);
        osc.stop(time + 0.12);
    },

    playBeep(time) {
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        
        osc.type = 'sine';
        osc.frequency.setValueAtTime(1500, time);
        osc.frequency.exponentialRampToValueAtTime(3000, time + 0.05);

        gain.gain.setValueAtTime(0, time);
        gain.gain.linearRampToValueAtTime(0.08, time + 0.01);
        gain.gain.exponentialRampToValueAtTime(0.0001, time + 0.2);

        osc.connect(gain);
        gain.connect(this.ctx.destination);

        osc.start(time);
        osc.stop(time + 0.2);
    },

    nextNote() {
        const secondsPerBeat = 60.0 / this.bpm;
        this.nextNoteTime += 0.25 * secondsPerBeat; // Add 1/16th note
        this.current16thNote++;
        if (this.current16thNote === 16) {
            this.current16thNote = 0;
        }
    },

    scheduleNote(beatNumber, time) {
        // manga_arps live_loop (minor pentatonic)
        const noteIndex = Math.floor(Math.random() * this.scale.length);
        this.playChip(time, this.scale[noteIndex]);

        // scifi_vibe at intervals (every 2 beats = every 8 16th notes)
        if (beatNumber % 8 === 0) {
            this.playBeep(time);
        }
    },

    scheduler() {
        while (this.nextNoteTime < this.ctx.currentTime + this.scheduleAheadTime) {
            this.scheduleNote(this.current16thNote, this.nextNoteTime);
            this.nextNote();
        }
        this.timerID = setTimeout(() => this.scheduler(), this.lookahead);
    },

    toggle() {
        this.init();
        if (this.isPlaying) {
            this.isPlaying = false;
            clearTimeout(this.timerID);
            this.stopProphet();
        } else {
            this.isPlaying = true;
            this.ctx.resume();
            this.nextNoteTime = this.ctx.currentTime;
            this.playProphet();
            this.scheduler();
        }
        return this.isPlaying;
    },

    // UI Reactivity (optional based on plan)
    setIntensity(score) {
        if (!this.ctx || !this.prophetFilter) return;
        // Open the filter for higher scores
        const freq = 400 + (score * 400);
        this.prophetFilter.frequency.setTargetAtTime(freq, this.ctx.currentTime, 0.5);
    },

    // Win effects: Glitch / Pitch-Slide
    glitchWin() {
        if (!this.ctx || !this.isPlaying) return;
        // Tempo drift
        const originalBpm = this.bpm;
        this.bpm = 300; 
        setTimeout(() => {
            this.bpm = 75;
            setTimeout(() => this.bpm = originalBpm, 500);
        }, 100);
    },

    pitchSlideWin() {
        if (!this.ctx || !this.prophet) return;
        const now = this.ctx.currentTime;
        this.prophet.osc1.frequency.exponentialRampToValueAtTime(10, now + 1.5);
        this.prophet.osc2.frequency.exponentialRampToValueAtTime(10, now + 1.5);
        this.prophet.gain.gain.exponentialRampToValueAtTime(0.0001, now + 1.5);
    }
};
