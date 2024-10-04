import torch
from flask import Flask, request, jsonify, render_template_string, send_file
import os
import subprocess
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import uuid

app = Flask(__name__)

def get_available_devices():
    devices = [('cpu', 'CPU')]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            devices.append((f'cuda:{i}', f'GPU {i}: {gpu_name}'))
    return devices

available_devices = get_available_devices()
default_device = 'cuda:0' if len(available_devices) > 1 else 'cpu'

def initialize_model(device):
    torch_dtype = torch.float16 if 'cuda' in device else torch.float32
    model_id = "openai/whisper-large-v3-turbo"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

model_cache = {}

HTML = '''
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Whisper Turbo 文字起こし</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100 p-8">
    <div class="max-w-4xl mx-auto bg-white p-8 rounded-lg shadow-md">
        <h1 class="text-3xl font-bold mb-6">Whisper Turbo 文字起こし</h1>
        <form id="uploadForm" class="mb-8">
            <div class="mb-4">
                <label for="deviceSelect" class="block text-sm font-medium text-gray-700 mb-2">デバイスを選択</label>
                <select id="deviceSelect" class="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md">
                    {% for device_id, device_name in available_devices %}
                    <option value="{{ device_id }}" {% if device_id == default_device %}selected{% endif %}>{{ device_name }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="mb-4">
                <label for="fileInput" class="block text-sm font-medium text-gray-700 mb-2">音声/動画ファイルを選択</label>
                <input type="file" id="fileInput" accept="audio/*,video/*" multiple required class="block w-full text-sm text-gray-500
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-full file:border-0
                    file:text-sm file:font-semibold
                    file:bg-blue-50 file:text-blue-700
                    hover:file:bg-blue-100
                ">
            </div>
            <div class="mb-4">
                <label for="languageSelect" class="block text-sm font-medium text-gray-700 mb-2">言語を選択（任意）</label>
                <select id="languageSelect" class="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md">
                    <option value="auto">自動検出</option>
                    <option value="ja">日本語</option>
                    <option value="en">英語</option>
                    <option value="zh">中国語</option>
                    <option value="ko">韓国語</option>
                    <option value="fr">フランス語</option>
                    <option value="de">ドイツ語</option>
                    <option value="es">スペイン語</option>
                </select>
            </div>
            <div class="mb-4">
                <label class="inline-flex items-center">
                    <input type="checkbox" id="translateCheck" class="form-checkbox h-5 w-5 text-blue-600">
                    <span class="ml-2 text-gray-700">英語に翻訳</span>
                </label>
            </div>
            <button type="submit" id="submitButton" class="w-full bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                文字起こし開始
            </button>
        </form>
        <div id="results" class="space-y-4"></div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('fileInput');
            const deviceSelect = document.getElementById('deviceSelect');
            const languageSelect = document.getElementById('languageSelect');
            const translateCheck = document.getElementById('translateCheck');
            const resultsDiv = document.getElementById('results');
            const submitButton = document.getElementById('submitButton');
            const formElements = document.querySelectorAll('#uploadForm input, #uploadForm select, #uploadForm button');

            if (fileInput.files.length === 0) {
                alert('少なくとも1つのファイルを選択してください');
                return;
            }

            // 無効化
            formElements.forEach(element => element.disabled = true);

            // クリア previous results
            resultsDiv.innerHTML = '';

            for (let file of fileInput.files) {
                const resultDiv = document.createElement('div');
                resultDiv.className = 'bg-gray-50 p-4 rounded-lg';
                resultDiv.innerHTML = `
                    <h3 class="font-bold mb-2">${file.name}</h3>
                    <p class="mb-2 text-gray-500">文字起こし中...</p>
                `;
                resultsDiv.appendChild(resultDiv);

                const formData = new FormData();
                formData.append('file', file);
                formData.append('device', deviceSelect.value);
                formData.append('language', languageSelect.value);
                formData.append('translate', translateCheck.checked);

                try {
                    const response = await fetch('/transcribe', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error('サーバーエラー');
                    }

                    const data = await response.json();
                    resultDiv.innerHTML = `
                        <h3 class="font-bold mb-2">${file.name}</h3>
                        <p class="mb-2">${data.transcription}</p>
                        <button onclick="copyToClipboard(this)" class="bg-green-500 hover:bg-green-700 text-white font-bold py-1 px-2 rounded mr-2">
                            コピー
                        </button>
                        <a href="/download/${data.id}" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-1 px-2 rounded">
                            ダウンロード
                        </a>
                    `;
                } catch (error) {
                    resultDiv.innerHTML = `
                        <h3 class="font-bold mb-2">${file.name}</h3>
                        <p class="text-red-500">処理中にエラーが発生しました: ${error.message}</p>
                    `;
                }
            }

            // 再有効化
            formElements.forEach(element => element.disabled = false);
        });

        function copyToClipboard(button) {
            const text = button.parentElement.querySelector('p').textContent;
            navigator.clipboard.writeText(text).then(() => {
                const originalText = button.textContent;
                button.textContent = 'コピーしました!';
                button.disabled = true;
                setTimeout(() => {
                    button.textContent = originalText;
                    button.disabled = false;
                }, 2000);
            }).catch(() => {
                alert('コピーに失敗しました');
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML, available_devices=available_devices, default_device=default_device)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "ファイルがありません"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "ファイルが選択されていません"}), 400
    
    device = request.form.get('device', default_device)
    language = request.form.get('language', 'auto')
    translate = request.form.get('translate', 'false').lower() == 'true'
    
    # アップロードされたファイルを保存
    uploads_dir = 'uploads'
    os.makedirs(uploads_dir, exist_ok=True)
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(uploads_dir, filename)
    file.save(file_path)
    
    # ファイルが動画かどうかチェック
    if file.content_type.startswith('video'):
        # FFmpegを使用してWAVに変換
        output_path = os.path.join('uploads', f'{uuid.uuid4()}.wav')
        ffmpeg_result = subprocess.run(['ffmpeg', '-y', '-i', file_path, '-acodec', 'pcm_s16le', '-ar', '16000', output_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ffmpeg_result.returncode != 0:
            os.remove(file_path)
            return jsonify({"error": "FFmpegによる音声抽出に失敗しました"}), 500
    else:
        output_path = file_path
    
    # 選択されたデバイスでモデルを初期化または取得
    if device not in model_cache:
        try:
            model_cache[device] = initialize_model(device)
        except Exception as e:
            if file.content_type.startswith('video'):
                os.remove(file_path)
                os.remove(output_path)
            return jsonify({"error": f"モデルの初期化に失敗しました: {str(e)}"}), 500
    pipe = model_cache[device]
    
    # Whisper Turboを使用して文字起こし
    generate_kwargs = {}
    if language != 'auto':
        generate_kwargs['language'] = language
    if translate:
        generate_kwargs['task'] = 'translate'
    
    try:
        result = pipe(output_path, return_timestamps=True, generate_kwargs=generate_kwargs)
    except Exception as e:
        if file.content_type.startswith('video'):
            os.remove(file_path)
            os.remove(output_path)
        return jsonify({"error": f"文字起こし中にエラーが発生しました: {str(e)}"}), 500
    
    # 文字起こし結果をファイルに保存
    transcription_id = str(uuid.uuid4())
    transcriptions_dir = 'transcriptions'
    os.makedirs(transcriptions_dir, exist_ok=True)
    transcription_path = os.path.join(transcriptions_dir, f'{transcription_id}.txt')
    with open(transcription_path, 'w', encoding='utf-8') as f:
        f.write(result["text"])
    
    # クリーンアップ
    os.remove(file_path)
    if file.content_type.startswith('video'):
        os.remove(output_path)
    
    return jsonify({"transcription": result["text"], "id": transcription_id})

@app.route('/download/<transcription_id>')
def download(transcription_id):
    transcription_path = os.path.join('transcriptions', f'{transcription_id}.txt')
    if not os.path.exists(transcription_path):
        return jsonify({"error": "ファイルが見つかりません"}), 404
    return send_file(transcription_path, as_attachment=True)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('transcriptions', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
