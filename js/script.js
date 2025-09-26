// --- LOAD MODEL ---
let session;
const labels = ["Glass", "Metal", "Plastic"]; // Sesuaikan dengan dls.vocab dari FastAI

async function loadModel() {
    try {
        session = await ort.InferenceSession.create("model_mobilenet_v3_small.onnx");
        console.log("ONNX model loaded");
        console.log("Input names:", session.inputNames);
        console.log("Output names:", session.outputNames);
    } catch (err) {
        console.error("Gagal load model:", err);
    }
}
loadModel();


// --- UPLOAD PHOTO / GALLERY ---
document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById('preview');
            preview.src = e.target.result;
            preview.style.display = "block"; 
            preview.onload = () => runInference(preview);
        };
        reader.readAsDataURL(file);
    }
});


async function runInference(imgElement) {
    if (!session) {
        document.getElementById("result").innerText = "Process in model ...";
        return;
    }

    try {
        const start = performance.now();

        // --- Preprocessing image ---
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        canvas.width = 224;
        canvas.height = 224;
        ctx.drawImage(imgElement, 0, 0, 224, 224);

        const imageData = ctx.getImageData(0, 0, 224, 224);
        const { data } = imageData;

        const float32Data = new Float32Array(3 * 224 * 224);

        for (let i = 0; i < 224 * 224; i++) {
            float32Data[i] = (data[i * 4] / 255 - 0.485) / 0.229;       // R
            float32Data[i + 224 * 224] = (data[i * 4 + 1] / 255 - 0.456) / 0.224; // G
            float32Data[i + 2 * 224 * 224] = (data[i * 4 + 2] / 255 - 0.406) / 0.225; // B
        }

        const inputTensor = new ort.Tensor("float32", float32Data, [1, 3, 224, 224]);
        const feeds = { [session.inputNames[0]]: inputTensor };

        console.log("Running inference...");
        const results = await session.run(feeds);
        const end = performance.now();

        const outputName = session.outputNames[0];
        const output = results[outputName].data;

        const argMax = output.indexOf(Math.max(...output));
        const prediction = labels[argMax];

        // --- Confidence ---
        const maxScore = output[argMax];
        const confidence = (maxScore * 100).toFixed(2);

        // --- Top-3 predictions ---
        const topK = 3;
        const sorted = output
            .map((score, idx) => ({ label: labels[idx], score }))
            .sort((a, b) => b.score - a.score)
            .slice(0, topK);

        const topKText = sorted
            .map(item => `${item.label}: ${(item.score * 100).toFixed(2)}%`)
            .join("\n");

        // --- Update HTML ---
        document.getElementById("result").innerHTML =
            `<b>Predicted:</b> ${prediction} (class ${argMax})<br>
             <b>Confidence:</b> ${confidence}%`;

        document.getElementById("inference-details").innerHTML =
            `<b>Top-${topK} Predictions:</b><br><pre>${topKText}</pre>
             <b>Inference Time:</b> ${(end - start).toFixed(2)} ms`;

        console.log("Inference results:", results);

    } catch (err) {
        console.error("Error saat inference:", err);
        document.getElementById("result").innerText = "Error saat inference, cek console.";
    }
}



// --- MOVING PAGE ---
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(div => {
        div.style.display = 'none';
    });
    document.getElementById(pageId).style.display = 'block';
}


// --- SELECT LANGUAGE ---
function setLanguage(lang) {
    document.querySelectorAll('.lang').forEach(el => el.style.display = 'none');
    document.querySelector('.lang-' + lang).style.display = 'block';
}
