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
const _simSlider = document.getElementById('simulationSlider');
const _simValue = document.getElementById('simulationValue');
const _badge = document.getElementById('turnBadge');
const _boardArea = document.getElementById('boardArea');
const _btnReset = document.getElementById('btnReset');

// Init
document.addEventListener("DOMContentLoaded", () => {
    updateStatsUI();
    loadModels();
    
    // Bind slider
    _simSlider.addEventListener('input', (e) => {
        _simValue.innerText = e.target.value;
    });

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

    // Check if column is full (top row is index 5 or 0 depending on logic, our row 5 is top visually but array wise 0 is top if Python logic is used?
    // Wait, Python logic: for row in reversed(range(6)): if board[row, col] == 0...
    // So row 5 is bottom, row 0 is top.
    
    // Check if row 0 (top) is empty
    if (board[0][c] !== 0) return;

    // Play move visually and internally
    const r = getLowestEmptyRow(c);
    if (r === -1) return;

    playMove(r, c, humanPlayer);

    if (checkWinResult(r, c, humanPlayer)) {
        endGame("You Win!");
        return;
    } else if (isDraw()) {
        endGame("It's a Draw!");
        return;
    }

    currentPlayer *= -1;
    updateTurnUI();
    triggerAiMove();
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
    _board.classList.add('disabled');
    
    try {
        const payload = {
            model: _modelSelect.value,
            board: board,
            current_player: currentPlayer,
            simulations: parseInt(_simSlider.value)
        };

        const res = await fetch('/api/move', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

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

        if (checkWinResult(row, col, currentPlayer)) {
            endGame("AI Wins!");
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
        let count = 1;
        for (let sign of [1, -1]) {
            let nr = r + dr * sign;
            let nc = c + dc * sign;
            
            while (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && board[nr][nc] === player) {
                count++;
                nr += dr * sign;
                nc += dc * sign;
            }
        }
        if (count >= 4) return true;
    }
    return false;
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

function endGame(message) {
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
