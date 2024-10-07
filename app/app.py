import torch
from flask import Flask, request, jsonify, render_template_string, send_file
import os
import subprocess
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import uuid
import json
import threading

app = Flask(__name__)

# タスク管理用の辞書とロック
tasks = {}
tasks_lock = threading.Lock()

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
        device=torch.device(device) if device.startswith('cuda') else -1,
    )
    return pipe

model_cache = {}

# HTML Template
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
            <!-- デバイス選択 -->
            <div class="mb-4">
                <label for="deviceSelect" class="block text-sm font-medium text-gray-700 mb-2">デバイスを選択</label>
                <select id="deviceSelect" class="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md">
                    {% for device_id, device_name in available_devices %}
                    <option value="{{ device_id }}" {% if device_id == default_device %}selected{% endif %}>{{ device_name }}</option>
                    {% endfor %}
                </select>
            </div>
            <!-- ポーリングオプション -->
            <div class="mb-4">
                <label class="inline-flex items-center">
                    <input type="checkbox" id="pollingCheck" class="form-checkbox h-5 w-5 text-blue-600">
                    <span class="ml-2 text-gray-700">ポーリングモードを有効にする</span>
                </label>
            </div>
            <!-- ファイル入力 -->
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
            <!-- 言語選択 -->
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
            <!-- 翻訳オプション -->
            <div class="mb-4">
                <label class="inline-flex items-center">
                    <input type="checkbox" id="translateCheck" class="form-checkbox h-5 w-5 text-blue-600">
                    <span class="ml-2 text-gray-700">英語に翻訳</span>
                </label>
            </div>
            <!-- チャンクアップロード設定 -->
            <div class="mb-4">
                <label class="inline-flex items-center">
                    <input type="checkbox" id="chunkUploadCheck" class="form-checkbox h-5 w-5 text-blue-600" checked>
                    <span class="ml-2 text-gray-700">チャンクアップロードを有効にする</span>
                </label>
            </div>
            <div class="mb-4" id="chunkSizeContainer">
                <label for="chunkSizeInput" class="block text-sm font-medium text-gray-700 mb-2">チャンクサイズ（MB）</label>
                <input type="number" id="chunkSizeInput" min="1" value="99" class="mt-1 block w-full pl-3 pr-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm">
            </div>
            <button type="submit" id="submitButton" class="w-full bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                文字起こし開始
            </button>
        </form>
        <div id="results" class="space-y-4"></div>
    </div>

    <script>
        document.getElementById('chunkUploadCheck').addEventListener('change', (e) => {
            const chunkSizeContainer = document.getElementById('chunkSizeContainer');
            chunkSizeContainer.style.display = e.target.checked ? 'block' : 'none';
        });

        document.getElementById('pollingCheck').addEventListener('change', (e) => {
            const isChecked = e.target.checked;
            // 必要に応じてポーリング有効時/無効時のUI変更を追加可能
        });

        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('fileInput');
            const deviceSelect = document.getElementById('deviceSelect');
            const languageSelect = document.getElementById('languageSelect');
            const translateCheck = document.getElementById('translateCheck');
            const chunkUploadCheck = document.getElementById('chunkUploadCheck');
            const chunkSizeInput = document.getElementById('chunkSizeInput');
            const pollingCheck = document.getElementById('pollingCheck');
            const resultsDiv = document.getElementById('results');
            const submitButton = document.getElementById('submitButton');
            const formElements = document.querySelectorAll('#uploadForm input, #uploadForm select, #uploadForm button');

            if (fileInput.files.length === 0) {
                alert('少なくとも1つのファイルを選択してください');
                return;
            }

            // フォーム要素を無効化
            formElements.forEach(element => element.disabled = true);

            // 前回の結果をクリア
            resultsDiv.innerHTML = '';

            const useChunkUpload = chunkUploadCheck.checked;
            const chunkSizeMB = parseInt(chunkSizeInput.value, 10) || 99;
            const chunkSize = chunkSizeMB * 1024 * 1024; // バイト単位

            const usePolling = pollingCheck.checked;

            for (let file of fileInput.files) {
                const resultDiv = document.createElement('div');
                resultDiv.className = 'bg-gray-50 p-4 rounded-lg';
                resultDiv.innerHTML = `
                    <h3 class="font-bold mb-2">${file.name}</h3>
                    <p class="mb-2 text-gray-500">文字起こし中...</p>
                `;
                resultsDiv.appendChild(resultDiv);

                try {
                    if (usePolling) {
                        let taskId;
                        if (useChunkUpload) {
                            const finalResponse = await uploadFileInChunksAsync(file, chunkSize, deviceSelect.value, languageSelect.value, translateCheck.checked);
                            taskId = finalResponse.task_id;
                        } else {
                            const response = await uploadFileAsync(file, deviceSelect.value, languageSelect.value, translateCheck.checked);
                            taskId = response.task_id;
                        }

                        // ポーリングを開始
                        pollTranscription(taskId, file.name, resultDiv);
                    } else {
                        let transcriptionData;
                        if (useChunkUpload) {
                            transcriptionData = await uploadFileInChunks(file, chunkSize, deviceSelect.value, languageSelect.value, translateCheck.checked);
                        } else {
                            transcriptionData = await uploadFile(file, deviceSelect.value, languageSelect.value, translateCheck.checked);
                        }

                        resultDiv.innerHTML = `
                            <h3 class="font-bold mb-2">${file.name}</h3>
                            <p class="mb-2">${transcriptionData.transcription}</p>
                            <button onclick="copyToClipboard(this)" class="bg-green-500 hover:bg-green-700 text-white font-bold py-1 px-2 rounded mr-2">
                                コピー
                            </button>
                            <a href="/download/${transcriptionData.id}" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-1 px-2 rounded">
                                ダウンロード
                            </a>
                        `;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `
                        <h3 class="font-bold mb-2">${file.name}</h3>
                        <p class="text-red-500">処理中にエラーが発生しました: ${error.message}</p>
                    `;
                }
            }

            // フォーム要素を再有効化
            formElements.forEach(element => element.disabled = false);
        });

        async function uploadFile(file, device, language, translate) {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('device', device);
            formData.append('language', language);
            formData.append('translate', translate);

            const response = await fetch('/transcribe', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'サーバーエラー');
            }

            return await response.json();
        }

        async function uploadFileAsync(file, device, language, translate) {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('device', device);
            formData.append('language', language);
            formData.append('translate', translate);
            formData.append('polling', 'true');

            const response = await fetch('/transcribe_async', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'サーバーエラー');
            }

            return await response.json();
        }

        async function uploadFileInChunks(file, chunkSize, device, language, translate) {
            const totalChunks = Math.ceil(file.size / chunkSize);
            const fileId = generateUUID();

            for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
                const start = chunkIndex * chunkSize;
                const end = Math.min(start + chunkSize, file.size);
                const chunk = file.slice(start, end);

                const formData = new FormData();
                formData.append('file', chunk);
                formData.append('device', device);
                formData.append('language', language);
                formData.append('translate', translate);
                formData.append('fileId', fileId);
                formData.append('chunkIndex', chunkIndex);
                formData.append('totalChunks', totalChunks);

                const response = await fetch('/transcribe_chunk', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'サーバーエラー');
                }
            }

            // 全チャンクがアップロードされた後に、サーバーで再構築と文字起こしを行います
            const finalResponse = await fetch('/transcribe_finalize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ fileId })
            });

            if (!finalResponse.ok) {
                const errorData = await finalResponse.json();
                throw new Error(errorData.error || 'サーバーエラー');
            }

            return await finalResponse.json();
        }

        async function uploadFileInChunksAsync(file, chunkSize, device, language, translate) {
            const totalChunks = Math.ceil(file.size / chunkSize);
            const fileId = generateUUID();

            for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
                const start = chunkIndex * chunkSize;
                const end = Math.min(start + chunkSize, file.size);
                const chunk = file.slice(start, end);

                const formData = new FormData();
                formData.append('file', chunk);
                formData.append('device', device);
                formData.append('language', language);
                formData.append('translate', translate);
                formData.append('fileId', fileId);
                formData.append('chunkIndex', chunkIndex);
                formData.append('totalChunks', totalChunks);
                formData.append('polling', 'true');

                const response = await fetch('/transcribe_chunk_async', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'サーバーエラー');
                }
            }

            // 最終化
            const finalResponse = await fetch('/transcribe_finalize_async', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ fileId })
            });

            if (!finalResponse.ok) {
                const errorData = await finalResponse.json();
                throw new Error(errorData.error || 'サーバーエラー');
            }

            return await finalResponse.json();
        }

        function generateUUID() { // RFC4122 version 4 compliant UUID
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        }

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

        async function pollTranscription(taskId, filename, resultDiv) {
            resultDiv.innerHTML = `
                <h3 class="font-bold mb-2">${filename}</h3>
                <p class="mb-2 text-gray-500">文字起こし中...</p>
            `;

            const pollInterval = 5000; // ポーリングレートの設定

            const intervalId = setInterval(async () => {
                try {
                    const response = await fetch(`/status/${taskId}`);
                    if (!response.ok) {
                        throw new Error('サーバーエラー');
                    }

                    const data = await response.json();
                    if (data.status === 'completed') {
                        clearInterval(intervalId);
                        resultDiv.innerHTML = `
                            <h3 class="font-bold mb-2">${data.filename}</h3>
                            <p class="mb-2">${data.transcription}</p>
                            <button onclick="copyToClipboard(this)" class="bg-green-500 hover:bg-green-700 text-white font-bold py-1 px-2 rounded mr-2">
                                コピー
                            </button>
                            <a href="/download/${data.id}" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-1 px-2 rounded">
                                ダウンロード
                            </a>
                        `;
                    } else if (data.status === 'error') {
                        clearInterval(intervalId);
                        resultDiv.innerHTML = `
                            <h3 class="font-bold mb-2">${data.filename}</h3>
                            <p class="text-red-500">処理中にエラーが発生しました: ${data.error}</p>
                        `;
                    }
                } catch (error) {
                    clearInterval(intervalId);
                    resultDiv.innerHTML = `
                        <h3 class="font-bold mb-2">${filename}</h3>
                        <p class="text-red-500">ポーリング中にエラーが発生しました: ${error.message}</p>
                    `;
                }
            }, pollInterval);
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML, available_devices=available_devices, default_device=default_device)

def process_transcription(file_path, device, language, translate, transcription_id, task_id=None):
    try:
        # ファイルが動画かどうかをチェック
        if file_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            # FFmpegを使用してWAVに変換
            output_wav_path = os.path.join('uploads', f'{uuid.uuid4()}.wav')
            ffmpeg_result = subprocess.run(['ffmpeg', '-y', '-i', file_path, '-acodec', 'pcm_s16le', '-ar', '16000', output_wav_path],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if ffmpeg_result.returncode != 0:
                raise Exception("FFmpegによる音声抽出に失敗しました")
            processing_path = output_wav_path
        else:
            processing_path = file_path

        # モデルの初期化または取得
        if device not in model_cache:
            model_cache[device] = initialize_model(device)
        pipe = model_cache[device]

        # 生成キーワードの準備
        generate_kwargs = {}
        if language != 'auto':
            generate_kwargs['language'] = language
        if translate:
            generate_kwargs['task'] = 'translate'

        # Whisper Turboを使用して文字起こし
        result = pipe(processing_path, return_timestamps=True, generate_kwargs=generate_kwargs)

        # 文字起こし結果をファイルに保存
        transcription_path = os.path.join('transcriptions', f'{transcription_id}.txt')
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(result["text"])

        # タスクステータスの更新
        if task_id:
            with tasks_lock:
                tasks[task_id]['status'] = 'completed'
                tasks[task_id]['transcription'] = result["text"]
                tasks[task_id]['id'] = transcription_id
                tasks[task_id]['filename'] = os.path.basename(file_path)

        # クリーンアップ
        os.remove(file_path)
        if processing_path != file_path:
            os.remove(processing_path)

    except Exception as e:
        if task_id:
            with tasks_lock:
                tasks[task_id]['status'] = 'error'
                tasks[task_id]['error'] = str(e)
                tasks[task_id]['filename'] = os.path.basename(file_path)
        # 追加のクリーンアップが必要な場合はここに記述

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # 同期的な文字起こし
    if 'file' not in request.files:
        return jsonify({"error": "ファイルがありません"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "ファイルが選択されていません"}), 400

    polling = request.form.get('polling', 'false').lower() == 'true'

    device = request.form.get('device', default_device)
    language = request.form.get('language', 'auto')
    translate = request.form.get('translate', 'false').lower() == 'true'

    # アップロードされたファイルを保存
    uploads_dir = 'uploads'
    os.makedirs(uploads_dir, exist_ok=True)
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(uploads_dir, filename)
    file.save(file_path)

    if polling:
        # 非同期処理
        transcription_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        with tasks_lock:
            tasks[task_id] = {
                'status': 'processing',
                'transcription': None,
                'id': None,
                'error': None,
                'filename': file.filename
            }

        thread = threading.Thread(target=process_transcription, args=(file_path, device, language, translate, transcription_id, task_id))
        thread.start()

        return jsonify({"task_id": task_id}), 202
    else:
        # 同期的な処理
        try:
            transcription_id = str(uuid.uuid4())
            process_transcription(file_path, device, language, translate, transcription_id)
            transcription_path = os.path.join('transcriptions', f'{transcription_id}.txt')
            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription_text = f.read()
            return jsonify({"transcription": transcription_text, "id": transcription_id})
        except Exception as e:
            return jsonify({"error": f"文字起こし中にエラーが発生しました: {str(e)}"}), 500

@app.route('/transcribe_async', methods=['POST'])
def transcribe_async():
    # ポーリング有効時の非同期文字起こし
    return transcribe()

@app.route('/transcribe_chunk', methods=['POST'])
def transcribe_chunk():
    # 同期的なチャンクアップロード処理
    if 'file' not in request.files:
        return jsonify({"error": "ファイルがありません"}), 400
    chunk = request.files['file']
    if chunk.filename == '':
        return jsonify({"error": "ファイルが選択されていません"}), 400

    device = request.form.get('device', default_device)
    language = request.form.get('language', 'auto')
    translate = request.form.get('translate', 'false').lower() == 'true'
    file_id = request.form.get('fileId')
    chunk_index = int(request.form.get('chunkIndex', 0))
    total_chunks = int(request.form.get('totalChunks', 1))

    if not file_id:
        return jsonify({"error": "fileIdが提供されていません"}), 400

    # チャンクを保存
    temp_dir = os.path.join('temp_chunks', file_id)
    os.makedirs(temp_dir, exist_ok=True)
    chunk_filename = os.path.join(temp_dir, f'chunk_{chunk_index}')
    chunk.save(chunk_filename)

    return jsonify({"message": f"チャンク {chunk_index + 1}/{total_chunks} を受信しました"}), 200

@app.route('/transcribe_chunk_async', methods=['POST'])
def transcribe_chunk_async():
    # 非同期的なチャンクアップロード処理
    if 'file' not in request.files:
        return jsonify({"error": "ファイルがありません"}), 400
    chunk = request.files['file']
    if chunk.filename == '':
        return jsonify({"error": "ファイルが選択されていません"}), 400

    device = request.form.get('device', default_device)
    language = request.form.get('language', 'auto')
    translate = request.form.get('translate', 'false').lower() == 'true'
    file_id = request.form.get('fileId')
    chunk_index = int(request.form.get('chunkIndex', 0))
    total_chunks = int(request.form.get('totalChunks', 1))

    if not file_id:
        return jsonify({"error": "fileIdが提供されていません"}), 400

    # チャンクを保存
    temp_dir = os.path.join('temp_chunks', file_id)
    os.makedirs(temp_dir, exist_ok=True)
    chunk_filename = os.path.join(temp_dir, f'chunk_{chunk_index}')
    chunk.save(chunk_filename)

    return jsonify({"message": f"チャンク {chunk_index + 1}/{total_chunks} を受信しました"}), 200

@app.route('/transcribe_finalize', methods=['POST'])
def transcribe_finalize():
    # 同期的な文字起こしの最終化
    return transcribe_finalize_helper(async_mode=False)

@app.route('/transcribe_finalize_async', methods=['POST'])
def transcribe_finalize_async():
    # 非同期的な文字起こしの最終化
    return transcribe_finalize_helper(async_mode=True)

def transcribe_finalize_helper(async_mode=False):
    data = request.get_json()
    if not data or 'fileId' not in data:
        return jsonify({"error": "fileIdが提供されていません"}), 400

    file_id = data['fileId']
    temp_dir = os.path.join('temp_chunks', file_id)
    if not os.path.exists(temp_dir):
        return jsonify({"error": "チャンクが見つかりません"}), 400

    try:
        # チャンクを番号順に並べて結合
        chunk_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('chunk_')], key=lambda x: int(x.split('_')[1]))
        if not chunk_files:
            return jsonify({"error": "チャンクがありません"}), 400

        reconstructed_file_path = os.path.join('uploads', f'{file_id}_reconstructed')
        with open(reconstructed_file_path, 'wb') as outfile:
            for chunk_file in chunk_files:
                chunk_path = os.path.join(temp_dir, chunk_file)
                with open(chunk_path, 'rb') as infile:
                    outfile.write(infile.read())

        # 拡張子から動画かどうかを判定
        is_video = False
        if reconstructed_file_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            is_video = True

        if is_video:
            # FFmpegを使用してWAVに変換
            output_wav_path = os.path.join('uploads', f'{uuid.uuid4()}.wav')
            ffmpeg_result = subprocess.run(['ffmpeg', '-y', '-i', reconstructed_file_path, '-acodec', 'pcm_s16le', '-ar', '16000', output_wav_path],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if ffmpeg_result.returncode != 0:
                return jsonify({"error": "FFmpegによる音声抽出に失敗しました"}), 500
            processing_path = output_wav_path
        else:
            processing_path = reconstructed_file_path

        if async_mode:
            # 非同期処理
            transcription_id = str(uuid.uuid4())
            task_id = str(uuid.uuid4())
            with tasks_lock:
                tasks[task_id] = {
                    'status': 'processing',
                    'transcription': None,
                    'id': None,
                    'error': None,
                    'filename': f'{file_id}_reconstructed'
                }

            device = request.form.get('device', default_device)
            language = request.form.get('language', 'auto')
            translate = request.form.get('translate', 'false').lower() == 'true'

            thread = threading.Thread(target=process_transcription, args=(processing_path, device, language, translate, transcription_id, task_id))
            thread.start()

            # チャンクの一時ファイルを削除
            for chunk_file in chunk_files:
                os.remove(os.path.join(temp_dir, chunk_file))
            os.rmdir(temp_dir)

            return jsonify({"task_id": task_id}), 202
        else:
            # 同期処理
            transcription_id = str(uuid.uuid4())
            process_transcription(processing_path, device=request.form.get('device', default_device),
                                  language=request.form.get('language', 'auto'),
                                  translate=request.form.get('translate', 'false').lower() == 'true',
                                  transcription_id=transcription_id)
            # チャンクの一時ファイルを削除
            for chunk_file in chunk_files:
                os.remove(os.path.join(temp_dir, chunk_file))
            os.rmdir(temp_dir)

            # 文字起こし結果を読み取って返す
            transcription_path = os.path.join('transcriptions', f'{transcription_id}.txt')
            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription_text = f.read()
            return jsonify({"transcription": transcription_text, "id": transcription_id})
    except Exception as e:
        return jsonify({"error": f"文字起こし中にエラーが発生しました: {str(e)}"}), 500

@app.route('/status/<task_id>')
def status(task_id):
    with tasks_lock:
        task = tasks.get(task_id)
        if not task:
            return jsonify({"error": "タスクが見つかりません"}), 404

        if task['status'] == 'completed':
            return jsonify({
                "status": "completed",
                "transcription": task['transcription'],
                "id": task['id'],
                "filename": task['filename']
            })
        elif task['status'] == 'error':
            return jsonify({
                "status": "error",
                "error": task['error'],
                "filename": task['filename']
            })
        else:
            return jsonify({
                "status": "processing",
                "filename": task['filename']
            })

@app.route('/download/<transcription_id>')
def download(transcription_id):
    transcription_path = os.path.join('transcriptions', f'{transcription_id}.txt')
    if not os.path.exists(transcription_path):
        return jsonify({"error": "ファイルが見つかりません"}), 404
    return send_file(transcription_path, as_attachment=True)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('transcriptions', exist_ok=True)
    os.makedirs('temp_chunks', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
