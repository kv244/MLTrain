/**
 * WinEffects Module: Randomized Cyberpunk Finishers
 */
const WinEffects = {
    activeEffect: null,

    triggerRandom(winner, winningLine) {
        // Find all tokens that ARE NOT in the winning line (losing tokens)
        const allTokens = document.querySelectorAll('.chip-1, .chip-2');
        const winningElements = winningLine.map(([r, c]) => document.getElementById(`spot-${r}-${c}`));
        const losingTokens = Array.from(allTokens).filter(token => !winningElements.includes(token));

        const effects = [
            () => this.digitalDissolve(winningElements, losingTokens),
            () => this.neonOverdrive(winningElements),
            () => this.systemBreach(winningElements, losingTokens)
        ];

        // Randomly pick one
        const chosen = effects[Math.floor(Math.random() * effects.length)];
        this.activeEffect = chosen;
        chosen();
    },

    digitalDissolve(winners, losers) {
        console.log("WinEffect: Digital Dissolve");
        losers.forEach(token => {
            // Hide the actual token
            token.classList.add('token-hidden');
            
            // Create particle explosion
            const rect = token.getBoundingClientRect();
            for (let i = 0; i < 8; i++) {
                const p = document.createElement('div');
                p.className = 'particle-hex';
                p.style.left = `${rect.left + rect.width / 2}px`;
                p.style.top = `${rect.top + rect.height / 2}px`;
                p.style.setProperty('--dx', `${(Math.random() - 0.5) * 150}px`);
                p.style.setProperty('--dy', `${(Math.random() - 0.5) * 150}px`);
                p.style.animation = `dissolve-particle ${0.5 + Math.random() * 1}s ease-out forwards`;
                document.body.appendChild(p);
                setTimeout(() => p.remove(), 1500);
            }
        });
    },

    neonOverdrive(winners) {
        console.log("WinEffect: Neon Overdrive");
        winners.forEach(w => w.classList.add('winning-overdrive'));
    },

    systemBreach(winners, losers) {
        console.log("WinEffect: System Breach");
        const overlay = document.getElementById('glitchOverlay');
        if (overlay) overlay.style.opacity = '1';
        document.body.classList.add('glitch-active');

        losers.forEach(token => {
            token.classList.add('breach-text');
            // Option to replace text or content if tokens had text, otherwise just style.
        });
    },

    reset() {
        // Clear all effects
        document.body.classList.remove('glitch-active');
        const overlay = document.getElementById('glitchOverlay');
        if (overlay) overlay.style.opacity = '0';

        document.querySelectorAll('.chip-1, .chip-2').forEach(t => {
            t.classList.remove('winning-overdrive', 'token-hidden', 'breach-text');
        });
        
        document.querySelectorAll('.particle-hex').forEach(p => p.remove());
        this.activeEffect = null;
    }
};
