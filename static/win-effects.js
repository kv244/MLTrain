/**
 * WinEffects Module: Randomized Cyberpunk Finishers (High-Fidelity Overhaul)
 */
const WinEffects = {

    triggerRandom(winner, winningLine) {
        // Find all tokens that ARE NOT in the winning line (losing tokens)
        const allTokens = document.querySelectorAll('.chip-1, .chip-2');
        const winningElements = winningLine.map(([r, c]) => document.getElementById(`spot-${r}-${c}`));
        const losingTokens = Array.from(allTokens).filter(token => !winningElements.includes(token));

        // Randomly pick one of the 3 upgraded effects
        const effects = [
            () => this.digitalDissolve(winningElements, losingTokens),
            () => this.neonOverdrive(winningElements),
            () => this.systemBreach(winningElements, losingTokens)
        ];

        const chosen = effects[Math.floor(Math.random() * effects.length)];
        chosen();
    },

    digitalDissolve(winners, losers) {
        console.log("WinEffect: Digital Dissolve 2.0");
        
        // Trigger Audio Sync (Pitch slide)
        if (typeof AudioEngine !== 'undefined') AudioEngine.pitchSlideWin();

        losers.forEach(token => {
            token.classList.add('token-hidden');
            
            // Create dense particle explosion (40 per token)
            const rect = token.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;

            for (let i = 0; i < 40; i++) {
                const p = document.createElement('div');
                p.className = 'particle-data';
                p.style.left = `${centerX}px`;
                p.style.top = `${centerY}px`;
                
                // Wide velocity jitter
                p.style.setProperty('--dx', `${(Math.random() - 0.5) * 300}px`);
                p.style.setProperty('--dy', `${(Math.random() - 0.5) * 300}px`);
                p.style.animationDelay = `${Math.random() * 0.4}s`;
                
                document.body.appendChild(p);
                setTimeout(() => p.remove(), 2000);
            }
        });
    },

    neonOverdrive(winners) {
        console.log("WinEffect: Neon Overdrive 2.0");
        
        // Trigger Screen Flash & Shockwave
        const flash = document.getElementById('screenFlash');
        const wave = document.getElementById('shockwave');
        
        if (flash) flash.classList.add('flash-trigger');
        if (wave) {
            // Origin at center of winning line
            const firstRect = winners[0].getBoundingClientRect();
            const lastRect = winners[winners.length - 1].getBoundingClientRect();
            const waveX = (firstRect.left + lastRect.left) / 2 + firstRect.width / 2;
            const waveY = (firstRect.top + lastRect.top) / 2 + firstRect.height / 2;
            
            wave.style.left = `${waveX}px`;
            wave.style.top = `${waveY}px`;
            wave.classList.add('wave-trigger');
        }

        winners.forEach(w => w.classList.add('winning-overdrive'));
        
        // Audio Glitch
        if (typeof AudioEngine !== 'undefined') AudioEngine.glitchWin();
    },

    systemBreach(winners, losers) {
        console.log("WinEffect: System Breach 2.0");
        
        const overlay = document.getElementById('glitchOverlay');
        if (overlay) overlay.style.opacity = '1';
        document.body.classList.add('glitch-active');
        document.querySelector('.app-container').classList.add('chromatic-aberration');

        // Audio Glitch (Intense drift)
        if (typeof AudioEngine !== 'undefined') AudioEngine.glitchWin();

        losers.forEach(token => {
            token.classList.add('breach-logic');
            // Fake logic text overhead if chips had labels, otherwise just glitch the color
        });
    },

    reset() {
        document.body.classList.remove('glitch-active');
        document.querySelector('.app-container')?.classList.remove('chromatic-aberration');
        
        const overlay = document.getElementById('glitchOverlay');
        const flash = document.getElementById('screenFlash');
        const wave = document.getElementById('shockwave');
        
        if (overlay) overlay.style.opacity = '0';
        if (flash) flash.classList.remove('flash-trigger');
        if (wave) wave.classList.remove('wave-trigger');

        document.querySelectorAll('.chip-1, .chip-2').forEach(t => {
            t.classList.remove('winning-overdrive', 'token-hidden', 'breach-logic');
        });
        
        document.querySelectorAll('.particle-data').forEach(p => p.remove());
    }
};
