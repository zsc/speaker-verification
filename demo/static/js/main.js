const socket = io();
let audioContext;
let processor;
let mediaStream;
let isRecording = false;

// UI Elements
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const logBox = document.getElementById('log-box');
const timeline = document.getElementById('event-timeline');
const heatmapDiv = document.getElementById('heatmap-container');

// Config Elements
const vadThresholdInput = document.getElementById('vad-threshold');
const vadThresholdVal = document.getElementById('vad-threshold-val');
const minSilenceInput = document.getElementById('min-silence');
const btnUpdateConfig = document.getElementById('btn-update-config');

// State
let config = {};

// Socket Events
socket.on('connect', () => {
    updateStatus('Connected', 'active');
});

socket.on('disconnect', () => {
    updateStatus('Disconnected', '');
});

socket.on('server_log', (data) => {
    log(data.msg);
});

socket.on('config_sync', (data) => {
    config = data;
    vadThresholdInput.value = config.vad_threshold;
    vadThresholdVal.innerText = config.vad_threshold;
    minSilenceInput.value = config.min_silence_duration_ms;
});

socket.on('vad_event', (data) => {
    const color = data.type === 'start' ? 'green' : 'red';
    log(`VAD Event: ${data.type} (prob: ${data.prob.toFixed(2)})`, color);
    addTimelineEvent(data.type === 'start' ? 'Speech Start' : 'Speech End', color);
});

socket.on('proc_event', (data) => {
    if (data.type === 'start') {
        updateStatus('Processing...', 'processing');
        addTimelineEvent('Proc Start', 'orange');
    } else {
        if (isRecording) updateStatus('Recording', 'recording');
        else updateStatus('Connected', 'active');
        addTimelineEvent('Proc End', 'orange');
    }
});

socket.on('matrix_update', (data) => {
    // data.matrix is 2D array, data.ids is list of IDs
    drawHeatmap(data.matrix, data.ids);
});

// Controls
btnStart.addEventListener('click', startRecording);
btnStop.addEventListener('click', stopRecording);
btnUpdateConfig.addEventListener('click', sendConfig);

vadThresholdInput.addEventListener('input', (e) => {
    vadThresholdVal.innerText = e.target.value;
});

// Functions
function updateStatus(text, className) {
    statusText.innerText = text;
    statusDot.className = 'status-indicator ' + (className ? 'status-' + className : '');
}

function log(msg, color) {
    const div = document.createElement('div');
    div.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
    if (color) div.style.color = color;
    logBox.appendChild(div);
    logBox.scrollTop = logBox.scrollHeight;
}

function addTimelineEvent(label, color) {
    const el = document.createElement('div');
    el.style.position = 'absolute';
    el.style.height = '100%';
    el.style.width = '2px';
    el.style.backgroundColor = color;
    el.style.right = '0';
    el.title = label;
    
    timeline.appendChild(el);
    
    // Animate moving left
    let pos = 0;
    const interval = setInterval(() => {
        pos += 1;
        el.style.right = pos + 'px';
        if (pos > timeline.offsetWidth) {
            clearInterval(interval);
            el.remove();
        }
    }, 50); // Speed
}

function drawHeatmap(matrix, ids) {
    const labels = ids.map(id => `Seg ${id}`);
    
    const data = [{
        z: matrix,
        x: labels,
        y: labels,
        type: 'heatmap',
        colorscale: 'Viridis'
    }];
    
    const layout = {
        title: 'Cosine Distance Matrix',
        annotations: [],
        margin: {t: 50, b: 50, l: 50, r: 50}
    };

    // Add text values
    for ( let i = 0; i < labels.length; i++ ) {
        for ( let j = 0; j < labels.length; j++ ) {
            const val = matrix[i][j].toFixed(3);
            const textColor = val < 0.5 ? 'white' : 'black';
            layout.annotations.push({
                xref: 'x1', yref: 'y1',
                x: labels[j], y: labels[i],
                text: val,
                font: {
                    family: 'Arial',
                    size: 12,
                    color: textColor
                },
                showarrow: false
            });
        }
    }

    Plotly.newPlot('heatmap-container', data, layout);
}

function sendConfig() {
    const newConfig = {
        vad_threshold: parseFloat(vadThresholdInput.value),
        min_silence_duration_ms: parseInt(minSilenceInput.value)
    };
    socket.emit('update_config', newConfig);
}

// Audio Handling
async function startRecording() {
    try {
        // Request 16kHz audio directly
        const constraints = { audio: { sampleRate: 16000, channelCount: 1 } };
        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        const source = audioContext.createMediaStreamSource(mediaStream);
        
        // Use ScriptProcessor for raw data access (BufferSize 4096 = ~256ms latency, maybe lower for VAD?)
        // Lower buffer size = more frequent updates but higher CPU load. 
        // Silero expects chunks. 512 is good.
        // ScriptProcessor buffer sizes must be power of 2: 256, 512, 1024, 2048, 4096...
        // 512 @ 16k = 32ms. Perfect for Silero.
        processor = audioContext.createScriptProcessor(512, 1, 1);
        
        processor.onaudioprocess = (e) => {
            if (!isRecording) return;
            const inputData = e.inputBuffer.getChannelData(0);
            // Send to server (need to copy because buffer is reused)
            // Convert Float32Array to Array for JSON/Socket.io compatibility or send raw buffer
            // Socket.io handles TypedArrays usually.
            socket.emit('audio_chunk', inputData.buffer); 
        };
        
        source.connect(processor);
        processor.connect(audioContext.destination); // Needed for Chrome to activate processor
        
        isRecording = true;
        btnStart.disabled = true;
        btnStop.disabled = false;
        updateStatus('Recording', 'recording');
        
    } catch (err) {
        log('Error starting audio: ' + err, 'red');
        console.error(err);
    }
}

function stopRecording() {
    isRecording = false;
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    if (processor) {
        processor.disconnect();
    }
    if (audioContext) {
        audioContext.close();
    }
    btnStart.disabled = false;
    btnStop.disabled = true;
    updateStatus('Connected', 'active');
}
