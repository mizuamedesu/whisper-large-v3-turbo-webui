# whisper-large-v3-turbo-webui

## 使用方法

Docker Composeでアプリケーションを起動：
```
docker-compose up --build
```

ブラウザで `http://localhost:5000` にアクセス

## 使用ライブラリ/ツール

- Whisper large-v3-turbo model: https://huggingface.co/openai/whisper-large-v3-turbo
- FFmpeg: https://ffmpeg.org/

## ライセンス

このプロジェクトは Apache License 2.0 の下で提供されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。

## 免責事項

このソフトウェアは「現状のまま」提供されており、明示または黙示を問わず、いかなる種類の保証も行いません。作者または著作権所有者は、契約行為、不法行為、またはそれ以外にかかわらず、ソフトウェアに起因または関連し、あるいはソフトウェアの使用またはその他の扱いによって生じる一切の請求、損害、その他の義務について何らの責任も負わないものとします。

## 謝辞

- OpenAI Whisper model:
  ```
  @misc{radford2022whisper,
    doi = {10.48550/ARXIV.2212.04356},
    url = {https://arxiv.org/abs/2212.04356},
    author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
    title = {Robust Speech Recognition via Large-Scale Weak Supervision},
    publisher = {arXiv},
    year = {2022},
    copyright = {arXiv.org perpetual, non-exclusive license}
  }
  ```

- FFmpeg: This software uses code of FFmpeg licensed under the LGPLv2.1 and its source can be downloaded from https://ffmpeg.org/
