let mediaRecorder;
let audioChunks = [];
let chunkIndex = 0;

document.getElementById("recordButton").addEventListener("click", startRecording);
document.getElementById("stopButton").addEventListener("click", stopRecording);

async function startRecording() {
    let stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    
    mediaRecorder.ondataavailable = function(event) {
        audioChunks.push(event.data);
        processAudioChunk(event.data);
    };

    mediaRecorder.start();

    document.getElementById("recordButton").disabled = true;
    document.getElementById("stopButton").disabled = false;
}

function stopRecording() {
    mediaRecorder.stop();

    document.getElementById("recordButton").disabled = false;
    document.getElementById("stopButton").disabled = true;
}

async function processAudioChunk(audioBlob) {
    let audio = await audioBlob.arrayBuffer();
    while(audio.byteLength % 2 !== 0){
        let extendBuffer = new ArrayBuffer(audio.byteLength + 1)
        let extendView = new Int8Array(extendBuffer)

        extendView.set(new Int8Array(audio))
        console.log("Не кратно двум")

        audio = extendBuffer;
    }
    let float32 = new Float32Array(audio)
    console.log(float32)

    let data = {
        "user_id": "user123",
        "timestamp": new Date().toISOString().replace(/T/, ' ').replace(/\..+/, ''),
        "audio": {
            "chunk_index": chunkIndex++,
            "content": Array.from(float32),
            "duration": audioBlob.size / 1000 // Примерная оценка длительности
        },
        "metadata": {
            "device": navigator.userAgent,
            "language": navigator.language
        },
        "is_final_chunk": false
    };

    console.log(float32, 'float32')
    sendAudioData(data);
}


function sendAudioData(data) {
    fetch('http://127.0.0.1:8898/proccess_audio', {
        method: 'POST',
        body: JSON.stringify(data),
        headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Methods': '*',
            'Access-Control-Allow-Credentials': 'true'
        },
    })
    .then((response)=>  {
        if(!response.ok) {
            throw new Error('Это не успех')
        }
         console.log('Success:', response.json());
         return response.json()
    })
    .catch(error => {
        console.log('Error:', error);
    });

}