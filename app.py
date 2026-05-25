import streamlit as st
import sqlite3
import json
from datetime import datetime, timedelta
import MeCab
import ipadic
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import japanize_matplotlib
import random
import google.generativeai as genai

# 🔑 Geminiの準備
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# 🗄️ データベースの準備（日記と分析結果を保存する箱を作る）
def init_db():
    conn = sqlite3.connect('mind_sphere.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            content TEXT,
            bg_color TEXT,
            word_colors TEXT,
            word_scores TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# 🎨 画面の基本設定
st.set_page_config(page_title="Mind-Sphere", page_icon="🧠", layout="centered")
st.title("🧠 Mind-Sphere — 脳内感情ログ")

# サイドメニューで「日記を書く」か「過去の分析を見る」かを選ぶ
menu = st.sidebar.selectbox("メニュー", ["📝 今日のお題・日記入力", "📊 脳内データ分析"])

# ==========================================
# 1. 日記入力ページ
# ==========================================
if menu == "📝 今日のお題・日記入力":
    st.subheader("今日の頭の中を言葉で表現してください")
    today_str = datetime.today().strftime('%Y-%m-%d')
    st.info(f"日付: {today_str}")
    
    user_text = st.text_area("何でも自由に書いて良いんだよ。", height=200)
    
    if st.button("脳内に保存して分析！"):
        if user_text.strip() == "":
            st.warning("一言でもいいから吐き出して見よう！")
        else:
            with st.spinner("思考を解剖中..."):
                # TF-IDF 計算
                tagger = MeCab.Tagger(ipadic.MECAB_ARGS)
                sentences = [s + "。" for s in user_text.replace('\n', '').split('。') if s]
                corpus = []
                for sentence in sentences:
                    node = tagger.parseToNode(sentence)
                    words = []
                    while node:
                        word = node.surface
                        hinshi = node.feature.split(',')[0]
                        if hinshi == "名詞" and len(word) > 1:
                            words.append(word)
                        node = node.next
                    if words:
                        corpus.append(" ".join(words))
                
                if not corpus:
                    st.error("分析できるキーワードが見つかりませんてした。もう少し具体的な名詞を入れてみて！")
                else:
                    vectorizer = TfidfVectorizer()
                    X = vectorizer.fit_transform(corpus)
                    words_list = vectorizer.get_feature_names_out()
                    scores = X.sum(axis=0).A1
                    word_scores = {words_list[i]: float(scores[i]) for i in range(len(words_list))}
                    
                    # Geminiで感情色付け
                    target_words_str = ", ".join(word_scores.keys())
                    prompt = f"""
                    以下のテキストの文脈を読み取り、抽出されたキーワードの感情を分析しろ。
                    【テキスト】: {user_text}
                    【キーワード一覧】: {target_words_str}
                    必ず以下のJSON形式のみを出力しろ。マークダウン等は一切含めるな。
                    {{
                        "bg_color": "情熱・焦りなら #3A1111、冷静・論理なら #11223A、癒やしなら #113A1B",
                        "word_colors": {{
                            "キーワード": "情熱・焦りなら #FF4B4B、冷静・段取りなら #4B8BFF、癒やし・休息なら #4BFF8B"
                        }}
                    }}
                    """
                    response = model.generate_content(prompt)
                    clean_json = response.text.replace('```json', '').replace('```', '').strip()
                    
                    try:
                        ai_data = json.loads(clean_json)
                        bg_color = ai_data.get("bg_color", "#111111")
                        word_colors = ai_data.get("word_colors", {})
                    except:
                        bg_color = "#111111"
                        word_colors = {w: "#CCCCCC" for w in word_scores.keys()}
                    
                    # データベースに保存（すでにある場合は上書き）
                    conn = sqlite3.connect('mind_sphere.db')
                    c = conn.cursor()
                    c.execute('''
                        INSERT OR REPLACE INTO entries (date, content, bg_color, word_colors, word_scores)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (today_str, user_text, bg_color, json.dumps(word_colors), json.dumps(word_scores)))
                    conn.commit()
                    conn.close()
                    
                    st.success("データベースへ格納完了！左メニューの『脳内データ分析』から確認できるよ！")

# ==========================================
# 2. データ分析ページ
# ==========================================
else:
    st.subheader("📊 蓄積された脳内マインドマップ")
    
    # 期間選択用のタブを作成！
    tab1, tab2, tab3, tab4 = st.tabs(["📌 1日（今日）", "📅 1週間（直近7日）", "🗓️ 1年（直近365日）", "🌌 すべての期間"])
    
    # データベースから全データ読み込み
    conn = sqlite3.connect('mind_sphere.db')
    c = conn.cursor()
    c.execute("SELECT date, bg_color, word_colors, word_scores FROM entries ORDER BY date DESC")
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        st.info("まだデータが蓄積されていません。まずは日記を入力してデータを増やしていこう！")
    else:
        # 期間に応じたデータフィルタリング関数
        def 描画システム(filtered_rows, title):
            if not filtered_rows:
                st.warning(f"{title}のデータがまだ足りないよ！")
                return
            
            # 複数日ある場合のデータ集計（数理マージ）
            merged_scores = {}
            merged_colors = {}
            bg_colors_list = []
            
            for row in filtered_rows:
                bg_colors_list.append(row[1])
                day_colors = json.loads(row[2])
                day_scores = json.loads(row[3])
                
                for word, score in day_scores.items():
                    merged_scores[word] = merged_scores.get(word, 0.0) + score
                    if word in day_colors:
                        merged_colors[word] = day_colors[word]
            
            # 最も多く登場した背景色を全体の背景色にする（多数決ロジック）
            final_bg = max(set(bg_colors_list), key=bg_colors_list.count) if bg_colors_list else "#111111"
            
            # お絵描き
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor(final_bg)
            ax.set_facecolor(final_bg)
            ax.axis('off')
            
            # 最大スコアでノーマライズして文字サイズを調整
            max_score = max(merged_scores.values()) if merged_scores else 1
            
            for word, score in merged_scores.items():
                # スコア比率に応じてサイズ決定（15〜50の間）
                fontsize = 15 + (score / max_score * 35)
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                color = merged_colors.get(word, "#CCCCCC")
                
                ax.text(x, y, word, fontsize=fontsize, color=color, 
                        alpha=0.9, fontweight='bold', ha='center', va='center')
            
            st.pyplot(fig)
            st.write(f"📊 分析対象のログ件数: {len(filtered_rows)} 件")

        # --- 各タブの処理 ---
        today_str = datetime.today().strftime('%Y-%m-%d')
        
        with tab1:
            st.markdown("### 今日の脳内ステータス")
            today_data = [r for r in rows if r[0] == today_str]
            描画システム(today_data, "今日")
            
        with tab2:
            st.markdown("### 直近1週間の脳内推移")
            one_week_ago = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
            week_data = [r for r in rows if r[0] >= one_week_ago]
            描画システム(week_data, "1週間")
            
        with tab3:
            st.markdown("### 過去1年間のマインドコア")
            one_year_ago = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
            year_data = [r for r in rows if r[0] >= one_year_ago]
            描画システム(year_data, "1年")
            
        with tab4:
            st.markdown("### 記録開始からの全脳内ログ")
            描画システム(rows, "全期間")