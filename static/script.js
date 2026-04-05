// Game State
const ROWS = 6;
const COLS = 7;
let board = [];
let currentPlayer = 1; // 1 = Human, -1 = AI (default if AI plays first)
let humanPlayer = 1;
let gameOver = false;

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

// Init
document.addEventListener("DOMContentLoaded", () => {
    updateStatsUI();
    loadModels();
    
    

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
    
    initBoard();
    clearAssessment();
    WinEffects.reset();
    
    _boardArea.classList.remove('hidden');
    _btnReset.classList.add('hidden');
    _board.classList.remove('disabled');
    
    updateTurnUI();

    // If AI is player 1, trigger AI move immediately
    if (humanPlayer !== currentPlayer) {
        triggerAiMove();
    }
}

async function handleColumnClick(c) {
    if (gameOver || currentPlayer !== humanPlayer) return;

    // Check if row 0 (top) is empty
    if (board[0][c] !== 0) return;

    // Disable board immediately during processing
    _board.classList.add('disabled');

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
    await fetchAssessment(boardBefore, c, humanPlayer); // FIX 13: pass current mover

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

async function fetchAssessment(prevBoard, move, movingPlayer) { // FIX 13: added param
    const sims = parseInt(_difficultySelect.value, 10);
    if (isNaN(sims) || sims < 1) return; // FIX 15: validate difficulty

    try {
        const res = await fetch('/api/assess', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                model: _modelSelect.value,
                board: prevBoard,
                move: move,
                current_player: movingPlayer, // FIX 13: use movingPlayer
                simulations: sims // FIX 15: use validated sims
            })
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            console.error("Assessment API error:", res.status, errData);
            return;
        }

        const data = await res.json();
        if (data.score) {
            showAssessment(data);
        }
    } catch (e) {
        console.error("Assessment failed due to network error", e);
    }
}

function showAssessment(data) {
    console.log("Showing assessment UI:", data);
    const container = document.getElementById('assessmentContainer');
    
    // FIX 14: Use textContent for server-supplied strings
    const badge = document.createElement('div');
    badge.className = 'assessment-badge';
    
    const starsDiv = document.createElement('div');
    starsDiv.className = 'stars';
    starsDiv.innerHTML = `
        ${'★'.repeat(data.score).split('').map(s => `<span class="star-filled">${s}</span>`).join('')}
        ${'☆'.repeat(5-data.score).split('').map(s => `<span class="star-empty">${s}</span>`).join('')}
    `;
    
    const commentDiv = document.createElement('div');
    commentDiv.className = 'comment-text';
    commentDiv.textContent = data.comment; // FIX 14: safe text injection
    
    badge.appendChild(starsDiv);
    badge.appendChild(commentDiv);
    
    container.innerHTML = '';
    container.appendChild(badge);
    
    // Highlight best move if it was a blunder (score <= 2)
    clearBestMoveHint();
    if (data.score <= 2) {
        const bestCol = document.querySelector(`.column[data-col="${data.best_move}"]`);
        if (bestCol) bestCol.classList.add('best-move-hint');
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

function playMove(r, c, player) {
    board[r][c] = player;
    const spot = document.getElementById(`spot-${r}-${c}`);
    // visually map to css classes
    if (player === 1) spot.classList.add('chip-1');
    else spot.classList.add('chip-2');
}

async function triggerAiMove() {
    const sims = parseInt(_difficultySelect.value, 10);
    if (isNaN(sims) || sims < 1) return; // FIX 15: validate difficulty

    _board.classList.add('disabled');
    
    try {
        const payload = {
            model: _modelSelect.value,
            board: board,
            current_player: currentPlayer,
            simulations: sims // FIX 15: use validated sims
        };

        const res = await fetch('/api/move', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            let errorMsg = "Server Error";
            try {
               const data = await res.json();
               errorMsg = data.error || data.description || `Error ${res.status}`;
            } catch(e) {
               errorMsg = `HTTP Error ${res.status}`;
            }
            
            if (res.status === 429) {
               endGame("Rate Limit Exceeded");
            } else {
               endGame(errorMsg);
            }
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

        playMove(row, col, currentPlayer);

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

    } catch (e) {
        console.error("Network error fetching AI move", e);
        endGame("Connection Lost");
    }
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
    // Check if top row is full
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
    } else {
        _badge.innerText = "AI is thinking...";
        _badge.className = "player-turn-badge turn-ai";
    }
}

function endGame(message, winningLine = null) {
    gameOver = true;
    _board.classList.add('disabled');
    
    _badge.innerText = message;
    _badge.className = "player-turn-badge"; 
    
    let winnerId = "draw";
    
    // Add a specific generic class based on win
    if (message.includes("You")) {
        _badge.classList.add('turn-p1');
        stats.player++;
        winnerId = "human";
    }
    else if (message.includes("AI")) {
        _badge.classList.add('turn-p2');
        stats.ai++;
        winnerId = "ai";
    }

    // Trigger randomized win effect if there is a winner
    if (winnerId !== "draw" && winningLine) {
        WinEffects.triggerRandom(winnerId, winningLine);
    }

    if (winnerId !== "draw" || message.includes("Draw")) {
        stats.games++;
        localStorage.setItem("c4_stats", JSON.stringify(stats));
        updateStatsUI();
        
        // Push telemetry
        fetch('/api/game_end', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                winner: winnerId,
                model: _modelSelect.value || "unknown"
            })
        }).catch(e => console.error("Telemetry log failed:", e));
    }
    
    _btnReset.classList.remove('hidden');
}
