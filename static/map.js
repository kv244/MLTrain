(function () {
    "use strict";
    var canvas = document.getElementById("map");
    var ctx = canvas.getContext("2d");
    var W = canvas.width, H = canvas.height;

    function project(lat, lon) {
        return [(lon + 180) / 360 * W, (90 - lat) / 180 * H];
    }

    function drawGrid() {
        ctx.clearRect(0, 0, W, H);
        ctx.strokeStyle = "rgba(0,229,255,0.10)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (var lon = -180; lon <= 180; lon += 30) {
            var x = project(0, lon)[0];
            ctx.moveTo(x, 0); ctx.lineTo(x, H);
        }
        for (var lat = -90; lat <= 90; lat += 30) {
            var y = project(lat, 0)[1];
            ctx.moveTo(0, y); ctx.lineTo(W, y);
        }
        ctx.stroke();
        ctx.strokeStyle = "rgba(0,229,255,0.28)";
        ctx.beginPath();
        var eq = project(0, 0)[1];
        ctx.moveTo(0, eq); ctx.lineTo(W, eq);
        ctx.stroke();
    }

    function drawBubbles(rows, maxV) {
        rows.forEach(function (r) {
            var c = window.COUNTRY_CENTROIDS[r.country];
            if (!c) { return; }
            var p = project(c[0], c[1]);
            var rad = 8 + 40 * Math.sqrt(r.visitors / maxV);
            var g = ctx.createRadialGradient(p[0], p[1], 0, p[0], p[1], rad);
            g.addColorStop(0, "rgba(255,0,255,0.85)");
            g.addColorStop(1, "rgba(255,0,255,0.05)");
            ctx.fillStyle = g;
            ctx.beginPath(); ctx.arc(p[0], p[1], rad, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = "#fff";
            ctx.beginPath(); ctx.arc(p[0], p[1], 3.5, 0, Math.PI * 2); ctx.fill();
        });
        ctx.font = "600 22px Inter, sans-serif";
        ctx.fillStyle = "rgba(255,255,255,0.85)";
        rows.slice(0, 8).forEach(function (r) {
            var c = window.COUNTRY_CENTROIDS[r.country];
            if (!c) { return; }
            var p = project(c[0], c[1]);
            ctx.fillText(r.country, p[0] + 12, p[1] - 12);
        });
    }

    function fill(rows) {
        var tbody = document.querySelector("#list tbody");
        rows.forEach(function (r) {
            var tr = document.createElement("tr");
            [r.country, r.visitors, r.games].forEach(function (v) {
                var td = document.createElement("td");
                td.textContent = v;
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    }

    drawGrid();
    fetch("/api/country_stats")
        .then(function (r) { return r.json(); })
        .then(function (d) {
            var summary = document.getElementById("summary");
            if (d.error || !d.countries.length) {
                summary.textContent = "No data available right now.";
                return;
            }
            var total = d.countries.reduce(function (s, r) { return s + r.visitors; }, 0);
            var known = d.countries.filter(function (r) { return r.country !== "Unknown"; }).length;
            summary.textContent = total + " visitors from " + known + " countries";
            drawBubbles(d.countries, d.countries[0].visitors);
            fill(d.countries);
        })
        .catch(function () {
            document.getElementById("summary").textContent = "Could not load the data.";
        });
})();
