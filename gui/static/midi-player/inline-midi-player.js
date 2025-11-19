/**
 * Inline MIDI Player - Lightweight MIDI playback and visualization
 * Works with JSON note data extracted from Python
 */

class InlineMIDIPlayer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.midiData = null;
        this.isPlaying = false;
        this.currentTime = 0;
        this.duration = 0;
        this.audioContext = null;
        this.animationFrame = null;
        this.startTime = 0;
        this.tempo = 120;

        this.init();
    }

    init() {
        this.createUI();
        this.setupAudioContext();
    }

    createUI() {
        this.container.innerHTML = `
            <div class="midi-player-controls">
                <button id="playPauseBtn" class="control-btn" title="Play/Pause">
                    <span id="playIcon">▶️</span>
                </button>
                <div class="timeline-container">
                    <input type="range" id="timeline" min="0" max="1000" value="0" step="1">
                    <div class="time-display">
                        <span id="currentTime">0:00</span> / <span id="totalTime">0:00</span>
                    </div>
                </div>
            </div>
            <canvas id="pianoRoll" width="900" height="400"></canvas>
        `;

        this.canvas = document.getElementById('pianoRoll');
        this.ctx = this.canvas.getContext('2d');
        this.playPauseBtn = document.getElementById('playPauseBtn');
        this.timeline = document.getElementById('timeline');
        this.playIcon = document.getElementById('playIcon');

        // Set canvas size
        this.canvas.width = this.container.clientWidth || 900;
        this.canvas.height = 400;

        this.setupEventListeners();
    }

    setupEventListeners() {
        this.playPauseBtn.addEventListener('click', () => this.togglePlayPause());
        this.timeline.addEventListener('input', (e) => this.seek(parseFloat(e.target.value) / 1000));
        this.timeline.addEventListener('mousedown', () => {
            this.wasPaying = this.isPlaying;
            if (this.isPlaying) this.pause();
        });
        this.timeline.addEventListener('mouseup', () => {
            if (this.wasPlaying) this.play();
        });
    }

    setupAudioContext() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    loadMIDIData(midiData) {
        this.midiData = midiData;
        this.notes = midiData.notes || [];
        this.duration = midiData.duration || 0;
        this.tempo = midiData.tempo || 120;

        // Convert duration from beats to seconds
        this.durationSeconds = (this.duration / this.tempo) * 60;

        document.getElementById('totalTime').textContent = this.formatTime(this.durationSeconds);

        this.drawPianoRoll();
    }

    drawPianoRoll() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear canvas
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, width, height);

        if (!this.notes || this.notes.length === 0) {
            ctx.fillStyle = '#666';
            ctx.font = '16px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No notes to display', width / 2, height / 2);
            return;
        }

        // Find pitch range
        const minPitch = Math.min(...this.notes.map(n => n.pitch));
        const maxPitch = Math.max(...this.notes.map(n => n.pitch));
        const pitchRange = maxPitch - minPitch + 1;

        // Draw grid lines for piano keys
        const noteHeight = height / pitchRange;
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 0.5;

        for (let i = 0; i < pitchRange; i++) {
            const y = i * noteHeight;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();

            // Highlight white keys (C, D, E, F, G, A, B)
            const pitch = maxPitch - i;
            const pitchClass = pitch % 12;
            if ([0, 2, 4, 5, 7, 9, 11].includes(pitchClass)) {
                ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
                ctx.fillRect(0, y, width, noteHeight);
            }
        }

        // Draw notes
        const timeScale = width / this.duration;

        // Group notes by track for coloring
        const trackColors = [
            '#00d9ff',  // Cyan - usually melody
            '#ff006e',  // Pink - usually chords
            '#ffbe0b',  // Yellow
            '#8338ec',  // Purple
            '#06ffa5',  // Green
            '#ff5400'   // Orange
        ];

        this.notes.forEach(note => {
            const x = note.start * timeScale;
            const y = height - ((note.pitch - minPitch + 1) * noteHeight);
            const w = Math.max(2, (note.end - note.start) * timeScale);
            const h = noteHeight - 1;

            const color = trackColors[note.track % trackColors.length];
            const alpha = 0.6 + (note.velocity / 127) * 0.4;

            ctx.fillStyle = color;
            ctx.globalAlpha = alpha;
            ctx.fillRect(x, y, w, h);
            ctx.globalAlpha = 1.0;

            // Add border for better visibility
            ctx.strokeStyle = color;
            ctx.lineWidth = 0.5;
            ctx.strokeRect(x, y, w, h);
        });

        // Draw playhead
        if (this.currentTime > 0) {
            this.drawPlayhead();
        }
    }

    drawPlayhead() {
        const currentBeats = (this.currentTime / 60) * this.tempo;
        const x = (currentBeats / this.duration) * this.canvas.width;

        this.ctx.strokeStyle = '#ffffff';
        this.ctx.lineWidth = 2;
        this.ctx.globalAlpha = 0.8;
        this.ctx.beginPath();
        this.ctx.moveTo(x, 0);
        this.ctx.lineTo(x, this.canvas.height);
        this.ctx.stroke();
        this.ctx.globalAlpha = 1.0;
    }

    togglePlayPause() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    play() {
        this.isPlaying = true;
        this.playIcon.textContent = '⏸️';
        this.startTime = this.audioContext.currentTime - this.currentTime;
        this.animate();
    }

    pause() {
        this.isPlaying = false;
        this.playIcon.textContent = '▶️';
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }

    animate() {
        if (!this.isPlaying) return;

        this.currentTime = this.audioContext.currentTime - this.startTime;

        if (this.currentTime >= this.durationSeconds) {
            this.currentTime = 0;
            this.startTime = this.audioContext.currentTime;
            this.pause();
            return;
        }

        // Update UI
        const progress = (this.currentTime / this.durationSeconds) * 1000;
        this.timeline.value = progress;
        document.getElementById('currentTime').textContent = this.formatTime(this.currentTime);

        // Redraw with playhead
        this.drawPianoRoll();

        this.animationFrame = requestAnimationFrame(() => this.animate());
    }

    seek(normalizedValue) {
        this.currentTime = normalizedValue * this.durationSeconds;
        this.startTime = this.audioContext.currentTime - this.currentTime;
        this.drawPianoRoll();
        document.getElementById('currentTime').textContent = this.formatTime(this.currentTime);
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
}

// CSS for the player (injected as a style tag)
const playerStyles = `
    .midi-player-controls {
        display: flex !important;
        align-items: center !important;
        gap: 15px !important;
        padding: 15px !important;
        background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%) !important;
        border-radius: 8px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }

    .control-btn {
        width: 50px !important;
        height: 50px !important;
        border-radius: 50% !important;
        border: none !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-size: 20px !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }

    .control-btn:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.5) !important;
    }

    .control-btn:active {
        transform: scale(0.95) !important;
    }

    .timeline-container {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
    }

    #timeline {
        width: 100% !important;
        height: 6px !important;
        border-radius: 3px !important;
        outline: none !important;
        -webkit-appearance: none !important;
        background: #ddd !important;
        cursor: pointer !important;
    }

    #timeline::-webkit-slider-thumb {
        -webkit-appearance: none !important;
        width: 18px !important;
        height: 18px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        cursor: pointer !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2) !important;
    }

    #timeline::-moz-range-thumb {
        width: 18px !important;
        height: 18px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        cursor: pointer !important;
        border: none !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2) !important;
    }

    .time-display {
        font-size: 13px !important;
        color: #666 !important;
        font-family: 'Courier New', monospace !important;
        display: flex !important;
        justify-content: space-between !important;
        padding: 0 4px !important;
    }

    #pianoRoll {
        width: 100% !important;
        height: 400px !important;
        border-radius: 8px !important;
        display: block !important;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.3) !important;
    }
`;

// Inject styles immediately
if (typeof document !== 'undefined') {
    const styleSheet = document.createElement("style");
    styleSheet.textContent = playerStyles;
    document.head.appendChild(styleSheet);
}
