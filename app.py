#グリッチ機能を追加しやすい構造にする．

# Scanline, RGBずらし, ガウシアンノイズ
import streamlit as st
import numpy as np
from PIL import Image
import cv2
from io import BytesIO



####
#関数定義
####
# scanlineをつける関数
def add_scanline_glitch(
    current_img,
    seed_y = 42,
    seed_offset = 1000,
    num_scratched=500, 
    length_offsets=20
    ):
    height, width, _ = current_img.shape
    glitched = current_img.copy()
    rng_y = np.random.default_rng(seed=seed_y)
    rng_offset = np.random.default_rng(seed=seed_offset)
    for _ in range(num_scratched):
        y = rng_y.integers(0, height)        # ランダムな行（y座標）
        offset = rng_offset.integers(-length_offsets, length_offsets+1)         # 左右にどれだけずらすか
        glitched[y] = np.roll(glitched[y], offset, axis=0) # 横方向（axis=0）にロール
        #axis=1でnp.rollするとBGRがmod3で巡回置換される．
    return glitched


#RGBずらしをする関数
def apply_RGB_shift(
    img,
    r_shift=20,
    r_axis=1,
    g_shift=-3,
    g_axis=0,
    b_shift=10,
    b_axis=0
    ):
    copied_img = img.copy()
    height, width, _ = img.shape
    b, g, r = cv2.split(copied_img)
    #axis=0は横方向，axis=1は縦方向
    r = np.roll(r, r_shift, axis=r_axis)  # Rをaxis方向に
    g = np.roll(g, g_shift, axis=g_axis) # Gをaxis方向に
    b = np.roll(b, b_shift, axis=b_axis) # Bをaxis方向に
    glitched = cv2.merge([b, g, r])
    return glitched


#ガウシアンノイズ関数
def add_gaussian_noise(img, 
                       mean=0, 
                       std=25):
    noisy = img.astype(np.float32)
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy += noise
    noisy = np.clip(noisy, 0, 255)  # 値を0〜255に制限
    return noisy.astype(np.uint8)
######



def apply_binary_glitch_from_qiita(img, glitch_amount=100):
    """
    JPEGの画像データ部分のみをバイナリレベルで書き換えるグリッチ関数
    """
    pil_img = Image.fromarray(img)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=50)
    jpeg_data = bytearray(buffer.getvalue())

    # SOS (Start of Scan) を探す
    sos_marker = b'\xff\xda'
    eoi_marker = b'\xff\xd9'
    sos_index = jpeg_data.find(sos_marker)
    eoi_index = jpeg_data.find(eoi_marker)

    if sos_index == -1 or eoi_index == -1:
        return img  # マーカーが見つからなければスキップ

    start = sos_index + 2  # SOSマーカーの直後から
    end = eoi_index        # EOIの直前まで

    # 書き換え対象部分のみランダムに改変
    for _ in range(glitch_amount):
        i = np.random.randint(start, end)
        jpeg_data[i] = np.random.randint(0, 256)

    # 書き換えたJPEGバイナリを読み込み
    try:
        glitched_img = Image.open(BytesIO(jpeg_data)).convert("RGB")
    except Exception:
        return img  # 壊れたら元画像を返す

    return np.array(glitched_img)

####


def apply_binary_glitch_strict(img, glitch_amount=300, seed=42):
    """
    JPEGファイルの実イメージ部（FFDA〜FFD9の間）だけをシード付きでランダムに破壊する
    """
    # NumPy配列 → PIL画像に変換
    pil_img = Image.fromarray(img)

    # PIL → JPEGバイナリ化
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=50)
    jpeg_bytes = bytearray(buffer.getvalue())

    # SOS/EOIマーカー探索
    sos = b'\xff\xda'
    eoi = b'\xff\xd9'
    sos_index = jpeg_bytes.find(sos)
    eoi_index = jpeg_bytes.find(eoi)

    if sos_index == -1 or eoi_index == -1 or sos_index > eoi_index:
        return img  # エラー時は元画像

    # ランダム生成器（シード付き）
    rng = np.random.default_rng(seed)

    # 実画像部の範囲（FFDA直後〜FFD9直前）
    start = sos_index + 2
    end = eoi_index

    # glitch_amount 分ランダムに改変
    for _ in range(glitch_amount):
        idx = rng.integers(start, end)
        jpeg_bytes[idx] = rng.integers(0, 256)

    # 読み込みチャレンジ
    try:
        glitched_img = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
    except Exception:
        return img

    return np.array(glitched_img)


#############3
#パラメーター関数の設定
#######33


def scanline_params():
    seed_y = st.slider("ずらす行のランダムさ(シード値)", 0, 1000, 42)
    seed_offset = st.slider("ずらしの大きさのランダムさ(シード値)", 0, 1000, 1000)
    num_scratched = st.slider("ひっかき線の数", 0, 5000, 500)
    length_offsets = st.slider("ずらし量の範囲", 1, 500, 20)
    return {
        "seed_y": seed_y, 
        "seed_offset": seed_offset, 
        "num_scratched": num_scratched, 
        "length_offsets": length_offsets
    }


def apply_RGB_shift_params():
    length_of_shift=500
    r_shift=  st.slider("Rチャンネルのずらし量", 1, length_of_shift, 10)
    r_axis =  st.radio("R色のずらし軸を選んでください", [0, 1], index=0,  format_func=lambda x: "横" if x == 0 else "縦")
    g_shift =  st.slider("Gチャンネルのずらし量", 1, length_of_shift, 10)
    g_axis =  st.radio("G色のずらし軸を選んでください", [0, 1], index=1, format_func=lambda x: "横" if x == 0 else "縦")
    b_shift =  st.slider("Bチャンネルのずらし量", 1, length_of_shift, 10)
    b_axis =  st.radio("B色のずらし軸を選んでください", [0, 1], index=0, format_func=lambda x: "横" if x == 0 else "縦")
    return {
        "r_shift": r_shift,
        "r_axis": r_axis,
        "g_shift":g_shift,
        "g_axis": g_axis,
        "b_shift": b_shift,
        "b_axis":b_axis
    }


def add_gaussian_noise_params():
    mean = st.slider("平均の値", -10, 10, 0)
    std = st.slider("分散の値", 0, 1000, 25)
    return {
        "mean":mean,
        "std":std
    }


def apply_binary_glitch_from_qiita_params():
    glitch_amount = st.slider("書き換えるバイト数", 0, 10000, 300)
    return {
        "glitch_amount": glitch_amount
    }

def apply_binary_glitch_strict_params():
    glitch_amount = st.slider("グリッチするバイト数", 0, 10000, 300)
    seed = st.slider("乱数シード", 0, 10000, 42)
    return {
        "glitch_amount": glitch_amount,
        "seed": seed
    }




######
######


######
#辞書にグリッチ関数と，パラメター関数を登録
###

GLITCH_FUNCTIONS = {
    "スキャンライン": add_scanline_glitch,
     "RGBずらし": apply_RGB_shift, 
     "ガウシアンノイズ": add_gaussian_noise,
     "JPEGバイナリグリッチ": apply_binary_glitch_from_qiita,  # ← 追加
     "JPEGバイナリグリッチ試作２" :apply_binary_glitch_strict
}

PARAM_FUNCTIONS = {
    "スキャンライン": scanline_params,
     "RGBずらし": apply_RGB_shift_params, 
     "ガウシアンノイズ": add_gaussian_noise_params, 
     "JPEGバイナリグリッチ": apply_binary_glitch_from_qiita_params,  # ← 追加
     "JPEGバイナリグリッチ試作２" :apply_binary_glitch_strict_params
}

#######
######

#streamlit: START
st.title("Y. Ishikiのグリッチアプリ!")
st.markdown("現在, 以下のグリッチを提供しています： \n- " + "  \n- ".join(GLITCH_FUNCTIONS.keys()))

#エフェクトを選ぶ
effect = st.selectbox("適用するグリッチ", list(GLITCH_FUNCTIONS.keys()))



#サンプル画像の説明と読み込み
st.write("最初はサンプル画像を表示します。画像をアップロードすると上書きされます。")

#サンプル画像読み込み
default_img = Image.open("sample.jpg").convert("RGB")



#ファイルアップローダー
uploaded_file = st.file_uploader("画像を選んでください", type=["jpg", "jpeg", "png"])

# アップロードされた画像があればそれを使い、なければサンプル画像
if uploaded_file is not None:
    current_img = Image.open(uploaded_file).convert("RGB")
    caption = "アプロードされた画像"
else:
    current_img = default_img
    caption = "デフォルト画像"


# パラメータ調整スライダー


# パラメータUIの表示と取得
params = PARAM_FUNCTIONS[effect]()  # ← sliderとかを表示しつつ値取得

# グリッチ適用
img_np = np.array(current_img)
glitched_np = GLITCH_FUNCTIONS[effect](img_np, **params)
glitched_img = Image.fromarray(glitched_np)


#session_stateに画像追加

st.session_state["glitched_img"] = glitched_img


# 元画像とグリッチ画像を並べて表示
col1, col2, = st.columns(2)

with col1:
    st.image(current_img, caption="アップロードされた画像", use_container_width=True)

with col2:
    st.image(st.session_state["glitched_img"], caption=f"{effect}適用後の画像", use_container_width=True)

#st.session_stateに画像を保存して以下の部分で画像が生成された場合，つまりがセッションにglitched_imgがある場合に
#保存ボタンを表示し続けるし，タブで選択しても初期化されない．
#Streamlitはタブを動かしたりするたびに上から下まで再読み込みが起こる.



# セッションに画像があるときのみ、保存UIを表示
# 保存
if "glitched_img" in st.session_state:
    file_format = st.selectbox("保存形式を選んでください", ["PNG", "JPEG"])
    ext_map = {
    "PNG": "png",
    "JPEG": "jpg"
    }
    # セッションから画像を取り出す
    glitched_img = st.session_state["glitched_img"]
    # 保存ボタン（←追加！）
    buffer = BytesIO()
    if file_format == "JPEG":
        glitched_img.convert("RGB").save(buffer, format=file_format)
    else:
        glitched_img.save(buffer, format=file_format)
    buffer.seek(0)
    st.download_button(
        label="グリッチされた画像を保存する",
        data=buffer,
        file_name=f"glitched_image.{ext_map[file_format]}",
        mime=f"image/{ext_map[file_format]}"
    )

