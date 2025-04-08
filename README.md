# glitch-app
Y. Ishiki's Glitch Image Editor
このアプリは、画像にさまざまなグリッチエフェクト（スキャンライン、RGBずらし、ガウシアンノイズ）をリアルタイムで適用できるStreamlit製のWebアプリです。
本プロジェクトは、
伊敷喜斗（Yoshito Ishiki）が、OpenAIの ChatGPT を活用しながら開発したものです。
グリッチ処理のアイディア、Pythonによる実装方法、StreamlitでのUI設計、デプロイ方法などを対話的に相談・試行錯誤しながら構築しました。

特徴
画像をアップロードして、エフェクトを自由に調整
グリッチ後の画像はその場でプレビュー＆保存可能
グリッチ処理は NumPy / OpenCV / PIL を活用
Streamlit Cloud にデプロイ済み

実際のアプリ（Streamlit Cloud）
[こちらから体験できます](https://glitch-app-wnoi5cycaq9hvobbftidzq.streamlit.app)

使用技術
Python 3.x
NumPy / OpenCV / Pillow
Streamlit
GitHub / Streamlit Cloud
