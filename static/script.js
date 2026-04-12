// Game State
const ROWS = 6;
const COLS = 7;
let board = [];
let boardHistory = []; // Step 5: Undo stack
let currentPlayer = 1; // 1 = Human, -1 = AI (default if AI plays first)
let humanPlayer = 1;
let gameOver = false;
let moveCount = 0; // Track total moves

// Session metadata (populated by initWelcomeMessage after geo-IP resolves)
let sessionCountry = "";

// LocalStorage Persistence
let stats = JSON.parse(localStorage.getItem("c4_stats")) || { games: 0, player: 0, ai: 0 };

function updateStatsUI() {
    document.getElementById('statGames').innerText = stats.games;
    document.getElementById('statPlayerWins').innerText = stats.player;
    document.getElementById('statAiWins').innerText = stats.ai;
}

// DOM Elements
const _board = document.getElementById('c4Board');
const _modelSelect = document.getElementById('modelSelect');
const _difficultySelect = document.getElementById('difficultySelect');
const _badge = document.getElementById('turnBadge');
const _boardArea = document.getElementById('boardArea');
const _btnReset = document.getElementById('btnReset');
const _btnAudio = document.getElementById('btnToggleAudio');
const _btnHint = document.getElementById('btnHint');
const _btnUndo = document.getElementById('btnUndo');
const _heatmapLayer = document.getElementById('heatmapLayer');

// Init
document.addEventListener("DOMContentLoaded", () => {
    updateStatsUI();
    loadModels();
    
    // Audio Toggle
    if (_btnAudio) {
        _btnAudio.addEventListener('click', () => {
            const isPlaying = AudioEngine.toggle();
            syncAudioUI(isPlaying);
        });
    }

    // Welcome Message (Geo-IP)
    initWelcomeMessage();

    // Step 4: Hint button
    if (_btnHint) {
        _btnHint.addEventListener('click', () => getHint());
    }

    // Step 5: Undo button
    if (_btnUndo) {
        _btnUndo.addEventListener('click', () => undoMove());
    }

    // Start controls
    document.getElementById('btnPlayerFirst').addEventListener('click', () => startGame(1));
    document.getElementById('btnAiFirst').addEventListener('click', () => startGame(-1));
    _btnReset.addEventListener('click', () => startGame(humanPlayer));
});

async function loadModels() {
    try {
        const res = await fetch('/api/models');
        const data = await res.json();
        
        _modelSelect.innerHTML = '';
        if (data.models.length === 0) {
            _modelSelect.innerHTML = '<option value="">No models found (.onnx)</option>';
            return;
        }

        data.models.forEach(model => {
            const opt = document.createElement('option');
            opt.value = model;
            opt.innerText = model;
            _modelSelect.appendChild(opt);
        });
        
        // Fetch hardware info
        fetch('/api/info')
            .then(r => r.json())
            .then(info => {
                document.getElementById('hardwareBadge').innerText = info.device;
            })
            .catch(e => console.error("Info error:", e));
    } catch (e) {
        console.error("Failed to fetch models", e);
        _modelSelect.innerHTML = '<option value="">Server Error</option>';
    }
}

function initBoard() {
    _board.innerHTML = '';
    board = Array(ROWS).fill(0).map(() => Array(COLS).fill(0));
    boardHistory = []; // Clear history
    updateActionButtons();
    hideHeatmap();

    for (let c = 0; c < COLS; c++) {
        const colDiv = document.createElement('div');
        colDiv.classList.add('column');
        colDiv.dataset.col = c;
        colDiv.addEventListener('click', () => handleColumnClick(c));

        // We append from r=ROWS-1 down to 0. 
        // With column-reverse in CSS, this places r=5 (the bottom row in Python) at the visual bottom.
        for (let r = ROWS - 1; r >= 0; r--) {
            const spot = document.createElement('div');
            spot.classList.add('spot');
            spot.id = `spot-${r}-${c}`;
            colDiv.appendChild(spot);
        }
        _board.appendChild(colDiv);
    }
}

function startGame(human_role) {
    if (!_modelSelect.value) {
        alert("Please wait for models to load or ensure .onnx files exist.");
        return;
    }

    humanPlayer = human_role;
    currentPlayer = 1; // 1 always goes first in Connect4
    gameOver = false;
    moveCount = 0;
    
    initBoard();
    clearAssessment();
    WinEffects.reset();
    
    _boardArea.classList.remove('hidden');
    _btnReset.classList.add('hidden');
    _board.classList.remove('disabled');
    const kofi = document.getElementById('kofi-container');
    if (kofi) kofi.style.display = 'none';
    
    updateTurnUI();

    // Start music if not already playing (User gesture context from click)
    if (typeof AudioEngine !== 'undefined' && !AudioEngine.isPlaying) {
        const isPlaying = AudioEngine.toggle();
        syncAudioUI(isPlaying);
        
        // Auto-stop after 10 seconds
        setTimeout(() => {
            if (AudioEngine.isPlaying) {
                AudioEngine.toggle();
                syncAudioUI(false);
            }
        }, 10000);
    }

    // If AI is player 1, trigger AI move immediately
    if (humanPlayer !== currentPlayer) {
        triggerAiMove();
    }
}

async function handleColumnClick(c) {
    if (gameOver || currentPlayer !== humanPlayer) return;

    // Check if row 0 (top) is empty
    if (board[0][c] !== 0) return;

    // Save history BEFORE move
    saveHistory();

    // Disable board immediately during processing
    _board.classList.add('disabled');
    hideHeatmap();

    // PRE-MOVE: Save state for assessment
    const boardBefore = JSON.parse(JSON.stringify(board));

    // Play move visually and internally
    const r = getLowestEmptyRow(c);
    if (r === -1) {
        _board.classList.remove('disabled');
        return;
    }

    playMove(r, c, humanPlayer);

    // FETCH ASSESSMENT (Sequential to avoid server crash)
    const assessment = await fetchAssessment(boardBefore, c, humanPlayer);

    // Step 1.4: Use server-provided winning cells if available
    if (assessment && assessment.winning_cells && assessment.winning_cells.length > 0) {
        endGame("You Win!", assessment.winning_cells);
        return;
    }

    // Local fallback check
    const winningLine = checkWinResult(r, c, humanPlayer);
    if (winningLine) {
        endGame("You Win!", winningLine);
        return;
    } else if (isDraw()) {
        endGame("It's a Draw!");
        return;
    }

    currentPlayer *= -1;
    updateTurnUI();
    triggerAiMove();
}

async function fetchAssessment(prevBoard, move, movingPlayer) { 
    const sims = parseInt(_difficultySelect.value, 10);
    if (isNaN(sims) || sims < 1) return null;

    try {
        const res = await fetch('/api/assess', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                model: _modelSelect.value,
                board: prevBoard,
                move: move,
                current_player: movingPlayer, 
                simulations: sims 
            })
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            console.error("Assessment API error:", res.status, errData);
            return null;
        }

        const data = await res.json();
        if (data.score) {
            showAssessment(data);
        }
        return data;
    } catch (e) {
        console.error("Assessment failed due to network error", e);
        return null;
    }
}

function showAssessment(data) {
    const container = document.getElementById('assessmentContainer');
    
    // Create fragment for better perf
    const badge = document.createElement('div');
    badge.className = 'assessment-badge';
    
    const starsDiv = document.createElement('div');
    starsDiv.className = 'stars';
    
    for (let i = 0; i < data.score; i++) {
        const s = document.createElement('span');
        s.className = 'star-filled';
        s.textContent = '★';
        s.style.animationDelay = `${i * 0.15}s`;
        starsDiv.appendChild(s);
    }
    for (let i = 0; i < (5 - data.score); i++) {
        const s = document.createElement('span');
        s.className = 'star-empty';
        s.textContent = '☆';
        starsDiv.appendChild(s);
    }
    
    const commentDiv = document.createElement('div');
    commentDiv.className = 'comment-text';
    commentDiv.textContent = data.comment; 
    
    badge.appendChild(starsDiv);
    badge.appendChild(commentDiv);
    
    if (data.ai_quote) {
        const quoteDiv = document.createElement('div');
        quoteDiv.className = 'ai-quote';
        quoteDiv.textContent = data.ai_quote;
        badge.appendChild(quoteDiv);
    }
    
    container.innerHTML = '';
    container.appendChild(badge);
    
    if (typeof AudioEngine !== 'undefined' && AudioEngine.isPlaying) {
        AudioEngine.setIntensity(data.score);
        // Menace sting: position is critical (score 1–2 = very bad for the moving player)
        if (data.score <= 2) AudioEngine.playMenace();
    }

    clearBestMoveHint();
    if (data.score <= 2) {
        const bestCol = document.querySelector(`.column[data-col="${data.best_move}"]`);
        if (bestCol) bestCol.classList.add('best-move-hint');
    }
}

function syncAudioUI(isPlaying) {
    if (!_btnAudio) return;
    if (isPlaying) {
        _btnAudio.classList.add('active');
        _btnAudio.innerHTML = '<span class="icon">🔊</span> Soundtrack: ON';
    } else {
        _btnAudio.classList.remove('active');
        _btnAudio.innerHTML = '<span class="icon">♪</span> Soundtrack: OFF';
    }
}

function clearAssessment() {
    document.getElementById('assessmentContainer').innerHTML = '';
    clearBestMoveHint();
}

function clearBestMoveHint() {
    document.querySelectorAll('.column').forEach(c => c.classList.remove('best-move-hint'));
}

function getLowestEmptyRow(c) {
    for (let r = ROWS - 1; r >= 0; r--) {
        if (board[r][c] === 0) return r;
    }
    return -1;
}

// v1.8.0 — 2026-04-11
// playMove: added chip-fresh class (triggers CSS drop-in + sizzle animations
// defined in style.css) and calls spawnSparks() for particle burst on placement.
// chip-fresh is removed after 800 ms so the settled breathing-glow animation
// can take over without an explicit JS animation loop.
function playMove(r, c, player) {
    if (typeof AudioEngine !== 'undefined') AudioEngine.playSwoosh();
    createStardustTrail(c, r, player);

    board[r][c] = player;
    moveCount++;

    document.querySelectorAll('.latest-piece').forEach(el => el.classList.remove('latest-piece'));

    const spot = document.getElementById(`spot-${r}-${c}`);
    if (player === 1) spot.classList.add('chip-1');
    else if (player === -1) spot.classList.add('chip-2');

    // Random bounce heights so each drop lands slightly differently
    spot.style.setProperty('--bounce-h1', `-${(5 + Math.random() * 8).toFixed(1)}px`);
    spot.style.setProperty('--bounce-h2', `-${(1 + Math.random() * 4).toFixed(1)}px`);
    spot.classList.add('latest-piece', 'chip-fresh');
    spawnSparks(spot, player);
    setTimeout(() => spot.classList.remove('chip-fresh'), 800);
}

// v1.8.0 — 2026-04-11
// spawnSparks: .spot has overflow:hidden so pseudo-element sparks would be
// clipped. Instead we use getBoundingClientRect() to get the viewport centre
// of the spot, then append position:fixed <div class="spark"> elements directly
// to <body>. Each spark uses a CSS custom property --angle so the single
// @keyframes spark-fly rule handles all 8 directions.
function spawnSparks(spot, player) {
    const rect  = spot.getBoundingClientRect();
    const cx    = rect.left + rect.width  / 2;
    const cy    = rect.top  + rect.height / 2;
    const color = player === 1 ? '#ff2d55' : '#ffe700';
    const count = 8;
    for (let i = 0; i < count; i++) {
        const spark = document.createElement('div');
        spark.className = 'spark';
        spark.style.left = cx + 'px';
        spark.style.top  = cy + 'px';
        const angle = (360 / count) * i + (Math.random() * 22 - 11);
        spark.style.setProperty('--angle', `${angle}deg`);
        spark.style.background = color;
        spark.style.boxShadow  = `0 0 4px 1px ${color}`;
        document.body.appendChild(spark);
        setTimeout(() => spark.remove(), 620);
    }
}

function createStardustTrail(col, endRow, player) {
    const colEl = document.querySelector(`.column[data-col="${col}"]`);
    if (!colEl) return;

    // Ensure relative positioning for tracking
    colEl.style.position = 'relative';

    const colHeight = colEl.clientHeight;
    const endY = ((endRow + 0.5) / ROWS) * colHeight;

    const particleCount = 10;
    const duration = 400;

    for (let i = 0; i < particleCount; i++) {
        setTimeout(() => {
            const p = document.createElement('div');
            p.className = 'stardust-particle';
            p.style.cssText = `left: 50%; top: 0;`;
            
            colEl.appendChild(p);
            
            p.animate([
                { transform: 'translate(-50%, 0) scale(1)', opacity: 0.8 },
                { transform: `translate(-50%, ${endY}px) scale(1)`, opacity: 0.6, offset: 0.7 },
                { transform: `translate(calc(-50% + ${Math.random()*40-20}px), ${endY + 20}px) scale(0)`, opacity: 0 }
            ], {
                duration: 600,
                easing: 'cubic-bezier(0.25, 0.46, 0.45, 0.94)',
                fill: 'forwards'
            });

            setTimeout(() => p.remove(), 800);
        }, (i * (duration / particleCount)));
    }
}

// Step 4: Hint function
async function getHint() {
    if (gameOver || currentPlayer !== humanPlayer) return;
    
    _btnHint.disabled = true;
    _btnHint.innerText = "Thinking...";
    
    try {
        const sims = parseInt(_difficultySelect.value, 10);
        const res = await fetch('/api/move', { // Using move endpoint for "pure" AI best move
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                model: _modelSelect.value,
                board: board,
                current_player: humanPlayer,
                simulations: sims
            })
        });
        
        if (!res.ok) throw new Error("API failed");
        
        const data = await res.json();
        const bestColIdx = data.move;
        
        // Visual Hint
        clearBestMoveHint();
        const colEl = document.querySelector(`.column[data-col="${bestColIdx}"]`);
        if (colEl) {
            colEl.classList.add('best-move-hint');
            // Remove after 3s
            setTimeout(() => colEl.classList.remove('best-move-hint'), 3000);
        }
    } catch (e) {
        console.error("Hint failed", e);
    } finally {
        _btnHint.disabled = false;
        _btnHint.innerText = "Get Hint";
    }
}

// Step 5: Undo Move
function saveHistory() {
    boardHistory.push(JSON.parse(JSON.stringify(board)));
}

function undoMove() {
    if (boardHistory.length < 2 || gameOver || currentPlayer !== humanPlayer) return;

    // Pop the state from BEFORE AI's move AND the state from BEFORE player's move
    boardHistory.pop(); // AI state
    const previousState = boardHistory.pop();
    
    board = previousState;
    moveCount = Math.max(0, moveCount - 2); // Undo both player and AI move
    renderBoard();
    clearAssessment();
    hideHeatmap();
    updateActionButtons();
}

function renderBoard() {
    // Clear latest move highlight on full render/undo
    document.querySelectorAll('.latest-piece').forEach(el => el.classList.remove('latest-piece'));

    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const spot = document.getElementById(`spot-${r}-${c}`);
            spot.className = 'spot'; // Reset
            if (board[r][c] === 1) spot.classList.add('chip-1');
            else if (board[r][c] === -1) spot.classList.add('chip-2');
        }
    }
}

async function triggerAiMove() {
    const sims = parseInt(_difficultySelect.value, 10);
    if (isNaN(sims) || sims < 1) return;

    _board.classList.add('disabled');
    updateActionButtons();
    
    try {
        const payload = {
            model: _modelSelect.value,
            board: board,
            current_player: currentPlayer,
            simulations: sims 
        };

        const res = await fetch('/api/move', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            endGame(`Error ${res.status}`);
            return;
        }

        const data = await res.json();
        
        if (data.error) {
            alert("AI Error: " + data.error);
            endGame("System Error");
            return;
        }

        const col = data.move;
        const row = getLowestEmptyRow(col);
        
        if (row === -1) {
            console.error("AI tried to play in full column");
            return;
        }

        // Step 3: Show heatmap; trigger menace if AI is highly confident
        if (data.probs) {
            showHeatmap(data.probs);
            if (typeof AudioEngine !== 'undefined' && AudioEngine.isPlaying &&
                Math.max(...data.probs) > 0.65) {
                AudioEngine.playMenace();
            }
        }

        playMove(row, col, currentPlayer);

        // Step 1.4: Use server-provided winning cells
        if (data.winning_cells && data.winning_cells.length > 0) {
            endGame("AI Wins!", data.winning_cells);
            return;
        }

        const winningLine = checkWinResult(row, col, currentPlayer);
        if (winningLine) {
            endGame("AI Wins!", winningLine);
            return;
        } else if (isDraw()) {
            endGame("It's a Draw!");
            return;
        }

        currentPlayer *= -1;
        updateTurnUI();
        _board.classList.remove('disabled');
        updateActionButtons();

    } catch (e) {
        console.error("Network error fetching AI move", e);
        endGame("Connection Lost");
    }
}

// Step 3: Heatmap logic
function showHeatmap(probs) {
    _heatmapLayer.classList.remove('hidden');
    probs.forEach((p, i) => {
        const bar = document.getElementById(`bar-${i}`);
        if (bar) {
            bar.style.transform = `scaleY(${p * 2.5})`; // Scale for visibility
            bar.style.opacity = Math.max(0.1, p);
        }
    });
}

function hideHeatmap() {
    _heatmapLayer.classList.add('hidden');
}

function checkWinResult(r, c, player) {
    const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];
    for (let [dr, dc] of directions) {
        let line = [[r, c]];
        for (let sign of [1, -1]) {
            let nr = r + dr * sign;
            let nc = c + dc * sign;
            while (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && board[nr][nc] === player) {
                line.push([nr, nc]);
                nr += dr * sign;
                nc += dc * sign;
            }
        }
        if (line.length >= 4) return line;
    }
    return null;
}

function isDraw() {
    for (let c = 0; c < COLS; c++) {
        if (board[0][c] === 0) return false;
    }
    return true;
}

function updateTurnUI() {
    if (gameOver) return;
    
    if (currentPlayer === humanPlayer) {
        _badge.innerText = "Your Turn";
        _badge.className = "player-turn-badge turn-p1";
        document.documentElement.style.setProperty('--ui-glow', 'var(--neon-magenta)');
    } else {
        _badge.innerText = "AI is thinking...";
        _badge.className = "player-turn-badge turn-ai";
        document.documentElement.style.setProperty('--ui-glow', 'var(--neon-cyan)');
    }
}

function updateActionButtons() {
    if (_btnHint) _btnHint.disabled = gameOver || currentPlayer !== humanPlayer;
    if (_btnUndo) _btnUndo.disabled = gameOver || currentPlayer !== humanPlayer || boardHistory.length < 2;
}

function endGame(message, winningLine = null) {
    gameOver = true;
    _board.classList.add('disabled');
    updateActionButtons();
    
    _badge.innerText = `${message} in ${moveCount} moves`;
    _badge.className = "player-turn-badge"; 
    
    let winnerId = "draw";
    if (message.includes("You")) {
        _badge.classList.add('turn-p1');
        stats.player++;
        winnerId = "human";
    } else if (message.includes("AI")) {
        _badge.classList.add('turn-p2');
        stats.ai++;
        winnerId = "ai";
    }

    if (winnerId !== "draw" && winningLine) {
        // Step 1.3: Highlight winning cells
        winningLine.forEach(([r, c]) => {
            const spot = document.getElementById(`spot-${r}-${c}`);
            if (spot) spot.classList.add('winning-spot');
        });

        WinEffects.triggerRandom(winnerId, winningLine);
        
        // Auto-revert effects after 5 seconds but keep the pieces
        setTimeout(() => {
            WinEffects.reset();
            document.querySelectorAll('.winning-spot').forEach(el => el.classList.remove('winning-spot'));
        }, 5000);

        setTimeout(() => {
            _btnReset.classList.remove('hidden');
        }, 1500);
    } else {
        _btnReset.classList.remove('hidden');
    }

    if (winnerId !== "draw" || message.includes("Draw")) {
        stats.games++;
        localStorage.setItem("c4_stats", JSON.stringify(stats));
        updateStatsUI();
        
        fetch('/api/game_end', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                winner:  winnerId,
                model:   _modelSelect.value || "unknown",
                moves:   moveCount,
                country: sessionCountry
            })
        }).catch(e => console.error("Telemetry log failed:", e));

        const kofi = document.getElementById('kofi-container');
        if (kofi) kofi.style.display = 'block';
    }
}

// v1.8.0 (2026-04-11): geolocation now done browser-side so the lookup uses
// the client's real IP (server proxy always resolved to the server's US IP).
// geolocation-db.com is whitelisted in the CSP connect-src header.
// sessionStorage guard removed — toast shows on every page load (5 s, auto-dismiss).
async function initWelcomeMessage() {
    const toast = document.getElementById('welcomeToast');
    const msgEl = document.getElementById('welcomeMessage');
    if (!toast || !msgEl) return;

    toast.classList.remove('hidden');

    // Kick off all three requests in parallel
    const [geoRes, infoRes, statsRes] = await Promise.allSettled([
        fetch('https://geolocation-db.com/json/'),
        fetch('/api/geoip'),
        fetch('/api/stats')
    ]);

    let country = "the physical realm";
    if (geoRes.status === 'fulfilled' && geoRes.value.ok) {
        try {
            const geo = await geoRes.value.json();
            country = geo.country_name || country;
            sessionCountry = geo.country_name || "";
        } catch (_) {}
    }

    // Record session visit in BigQuery (INSERT new IP / UPDATE returning)
    fetch('/api/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ country: sessionCountry })
    }).catch(() => {});

    let wallpaperNote = '';
    if (infoRes.status === 'fulfilled' && infoRes.value.ok) {
        try {
            const info = await infoRes.value.json();
            const d = info.wallpaper_days_left;
            if (d !== null && d !== undefined) {
                wallpaperNote = ` · wallpaper renews in ${d} day${d === 1 ? '' : 's'}`;
            }
        } catch (_) {}
    }

    let globalNote = '';
    if (statsRes.status === 'fulfilled' && statsRes.value.ok) {
        try {
            const s = await statsRes.value.json();
            if (s.total_games !== null && s.total_games !== undefined) {
                globalNote = ` · ${s.total_games.toLocaleString()} games played globally`;
            }
        } catch (_) {}
    }

    msgEl.innerText = `Thank you for joining me from ${country}!${globalNote}${wallpaperNote}`;

    // Auto-dismiss after 5 seconds to match progress bar
    setTimeout(() => {
        toast.style.animation = 'toast-slide-out 0.5s forwards';
        setTimeout(() => toast.remove(), 500);
    }, 5000);
}
