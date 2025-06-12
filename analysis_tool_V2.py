# interactive_analysis_tool.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- 數據處理函式 ---
def load_and_clean_data(uploaded_file):
    """從上傳的檔案載入並清理數據，回傳一個處理好的DataFrame"""
    if uploaded_file is None:
        return None
    try:
        # 關鍵修正：從使用者告知的第三行開始讀取 (header=2)
        df = pd.read_csv(uploaded_file, header=2, thousands=',', low_memory=False)
        df.columns = df.columns.str.strip()
        
        # 假設時間欄位為 'Time'
        time_column = 'Time'
        if time_column not in df.columns:
            st.error(f"錯誤：在第3行中找不到名為 '{time_column}' 的時間欄位。")
            return None

        df[time_column] = pd.to_datetime(df[time_column], format='%H:%M:%S:%f', errors='coerce')
        df.dropna(subset=[time_column], inplace=True)
        
        if df.empty: return None
        
        start_time = df[time_column].iloc[0]
        df['Elapsed Time (s)'] = (df[time_column] - start_time).dt.total_seconds()
        return df
    except Exception as e:
        st.error(f"讀取或處理檔案時發生錯誤: {e}")
        return None

# --- 升級版！動態圖表繪製函式 ---
def generate_flexible_chart(df, left_col, right_col, x_limits, y_limits):
    """根據使用者選擇的欄位與範圍，動態產生圖表"""
    if df is None or left_col is None:
        return None

    # 將選擇的欄位轉為數值，忽略錯誤
    df_chart = df.copy()
    df_chart.loc[:, 'left_val'] = pd.to_numeric(df_chart[left_col], errors='coerce')
    if right_col:
        df_chart.loc[:, 'right_val'] = pd.to_numeric(df_chart[right_col], errors='coerce')

    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.title(f'{left_col} vs. {right_col if right_col else ""}', fontsize=16)
    
    # --- 繪製左Y軸 ---
    color = 'tab:blue'
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel(left_col, color=color, fontsize=12)
    ax1.plot(df_chart['Elapsed Time (s)'], df_chart['left_val'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', linewidth=0.5)

    # --- 繪製右Y軸 (如果使用者有選) ---
    if right_col:
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel(right_col, color=color, fontsize=12)
        ax2.plot(df_chart['Elapsed Time (s)'], df_chart['right_val'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

    # --- 套用使用者自訂的範圍 ---
    ax1.set_xlim(x_limits)
    ax1.set_ylim(y_limits['left'])
    if right_col:
        ax2.set_ylim(y_limits['right'])

    fig.tight_layout()
    return fig

# --- Streamlit 網頁應用程式介面 ---
st.set_page_config(layout="wide")
st.title("互動式熱功耗數據探索平台")

# 側邊欄
st.sidebar.header("控制面板")
uploaded_log_file = st.sidebar.file_uploader("1. 上傳 Log File (.csv)", type="csv")

df = load_and_clean_data(uploaded_log_file)

if df is not None:
    # 篩選出所有數值型的欄位作為選項
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if 'Elapsed Time (s)' in numeric_columns:
        numeric_columns.remove('Elapsed Time (s)')
    
    st.sidebar.header("圖表設定")
    # --- UI元件：下拉選單 ---
    left_y_axis = st.sidebar.selectbox("選擇左側Y軸變數", options=numeric_columns, index=numeric_columns.index('Miscellaneous-MSR Package Temperature(Degree C)') if 'Miscellaneous-MSR Package Temperature(Degree C)' in numeric_columns else 0)
    right_y_axis_options = [None] + numeric_columns
    right_y_axis = st.sidebar.selectbox("選擇右側Y軸變數 (可不選)", options=right_y_axis_options, index=right_y_axis_options.index('Power-Package Power(Watts)') if 'Power-Package Power(Watts)' in right_y_axis_options else 0)

    # --- UI元件：數字輸入框 ---
    st.sidebar.header("座標軸範圍設定")
    x_min = st.sidebar.number_input("X軸最小值 (秒)", value=0.0)
    x_max = st.sidebar.number_input("X軸最大值 (秒)", value=7200.0)
    
    st.sidebar.subheader(f"'{left_y_axis}' (左Y軸) 範圍")
    y1_min = st.sidebar.number_input("左Y軸最小值", value=40.0, format="%.2f")
    y1_max = st.sidebar.number_input("左Y軸最大值", value=120.0, format="%.2f")

    if right_y_axis:
        st.sidebar.subheader(f"'{right_y_axis}' (右Y軸) 範圍")
        y2_min = st.sidebar.number_input("右Y軸最小值", value=0.0, format="%.2f")
        y2_max = st.sidebar.number_input("右Y軸最大值", value=70.0, format="%.2f")
    else:
        # 如果沒有右Y軸，則用None作為預留位置
        y2_min, y2_max = (None, None)

    # 主畫面
    st.header("動態比較圖表")
    fig = generate_flexible_chart(df, left_y_axis, right_y_axis, (x_min, x_max), {'left': (y1_min, y1_max), 'right': (y2_min, y2_max)})
    
    if fig:
        st.pyplot(fig)
    else:
        st.warning("無法產生圖表，請確認選擇的欄位。")

else:
    st.info("請從左側側邊欄上傳您的 Log File 開始分析。")