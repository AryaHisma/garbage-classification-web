// --- LOAD MODEL ---
let session;
const labels = ["Plastic", "Glass", "Metal"]; // sesuaikan dengan dls.vocab dari FastAI

async function loadModel() {
    session = await ort.InferenceSession.create("model.onnx");
    console.log("ONNX model loaded");
}
loadModel();



// UPLOAD PHOTO / GALLERY
document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
        const preview = document.getElementById('preview');
        preview.src = e.target.result;
        preview.style.display = "block"; 
        // Panggil inferensi setelah gambar siap
        preview.onload = () => runInference(preview);
        };
        reader.readAsDataURL(file);
    }
    });



// --- RUN INFERENCE ---
async function runInference(imgElement) {
    if (!session) {
        document.getElementById("result").innerText = "Model belum siap...";
        return;
    }

    // Convert image ke tensor [1,3,224,224]
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = 224;
    canvas.height = 224;
    ctx.drawImage(imgElement, 0, 0, 224, 224);

    const imageData = ctx.getImageData(0, 0, 224, 224);
    const { data } = imageData;

    // Normalisasi [0..255] → [0..1]
    const float32Data = new Float32Array(3 * 224 * 224);
    for (let i = 0; i < 224 * 224; i++) {
        float32Data[i] = data[i * 4] / 255;       // R
        float32Data[i + 224 * 224] = data[i * 4 + 1] / 255; // G
        float32Data[i + 2 * 224 * 224] = data[i * 4 + 2] / 255; // B
    }

    const inputTensor = new ort.Tensor("float32", float32Data, [1, 3, 224, 224]);

    // Jalankan model
    const feeds = { input: inputTensor }; // "input" = nama input ONNX (cek pakai onnx.load)
    const results = await session.run(feeds);

    const output = results["output"].data; // "output" = nama output ONNX
    const argMax = output.indexOf(Math.max(...output));
    const prediction = labels[argMax];

    // Tampilkan hasil
    document.getElementById("result").innerText = 'Predicted: ${prediction} (class ${argMax})';
}

// MOVING PAGE
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(div => {
        div.style.display = 'none';
    });
    document.getElementById(pageId).style.display = 'block';
    };


// SELECT LANGUANGE
function setLanguage(lang) {
    document.querySelectorAll('.lang').forEach(el => el.style.display = 'none');
    document.querySelector('.lang-' + lang).style.display = 'block';
}