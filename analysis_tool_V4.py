# final_interactive_tool_v4.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- 數據處理函式 (最終修正版) ---
def load_and_clean_data(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        # 關鍵修正：從第一行讀取標題 (header=0)，不再跳過
        uploaded_file.seek(0) 
        df = pd.read_csv(uploaded_file, header=0, thousands=',', low_memory=False)
        
        df.columns = df.columns.str.strip()
        
        time_column = 'Time' 
        if time_column not in df.columns:
            st.error(f"錯誤：找不到名為 '{time_column}' 的時間欄位。")
            st.write("偵測到的欄位有：", df.columns.tolist())
            return None

        # 清理時間欄位中的不可見字元
        time_series = df[time_column].astype(str).str.strip()
        # 修正：部分log檔的時間格式在毫秒部分使用'.'，部分使用':'，此處統一替換
        time_series_cleaned = time_series.str.replace(r':(\d{3})$', r'.\1', regex=True)
        
        df['Time_Cleaned'] = pd.to_datetime(time_series_cleaned, format='%H:%M:%S.%f', errors='coerce')
        
        df.dropna(subset=['Time_Cleaned'], inplace=True)
        if df.empty:
            st.error("錯誤：清理後，所有行的時間格式依然無法解析或檔案為空。")
            return None
        
        start_time = df['Time_Cleaned'].iloc[0]
        df['Elapsed Time (s)'] = (df['Time_Cleaned'] - start_time).dt.total_seconds()
        return df
    except Exception as e:
        st.error(f"讀取或處理檔案時發生未知錯誤: {e}")
        return None

# --- 動態圖表繪製函式 ---
def generate_flexible_chart(df, left_col, right_col, x_limits, y_limits):
    if df is None or left_col is None: return None
    df_chart = df.copy()
    
    if x_limits:
        df_chart = df_chart[(df_chart['Elapsed Time (s)'] >= x_limits[0]) & (df_chart['Elapsed Time (s)'] <= x_limits[1])]
    
    df_chart.loc[:, 'left_val'] = pd.to_numeric(df_chart[left_col], errors='coerce')
    if right_col and right_col != 'None':
        df_chart.loc[:, 'right_val'] = pd.to_numeric(df_chart[right_col], errors='coerce')

    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.title(f'{left_col} {"& " + right_col if right_col and right_col != "None" else ""}', fontsize=16)
    color = 'tab:blue'
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel(left_col, color=color, fontsize=12)
    ax1.plot(df_chart['Elapsed Time (s)'], df_chart['left_val'], color=color)
    ax1.tick_params(axis='y', labelcolor=color); ax1.grid(True, linestyle='--', linewidth=0.5)

    if right_col and right_col != 'None':
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel(right_col, color=color, fontsize=12)
        ax2.plot(df_chart['Elapsed Time (s)'], df_chart['right_val'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        if y_limits.get('right'): ax2.set_ylim(y_limits['right'])

    if x_limits: ax1.set_xlim(x_limits)
    if y_limits.get('left'): ax1.set_ylim(y_limits['left'])
    fig.tight_layout()
    return fig

# --- Streamlit 網頁應用程式介面 ---
st.set_page_config(layout="wide")
st.title("互動式熱功耗數據探索平台")

st.sidebar.header("控制面板")
uploaded_log_file = st.sidebar.file_uploader("1. 上傳 Log File (.csv)", type="csv")

if uploaded_log_file is not None:
    df = load_and_clean_data(uploaded_log_file)
    
    if df is not None:
        st.sidebar.success("檔案載入成功！")
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if 'Elapsed Time (s)' in numeric_columns: numeric_columns.remove('Elapsed Time (s)')
        
        st.sidebar.header("圖表設定")
        left_y_axis = st.sidebar.selectbox("選擇左側Y軸變數", options=numeric_columns, index=numeric_columns.index('Miscellaneous-MSR Package Temperature(Degree C)') if 'Miscellaneous-MSR Package Temperature(Degree C)' in numeric_columns else 0)
        right_y_axis_options = ['None'] + numeric_columns
        right_y_axis = st.sidebar.selectbox("選擇右側Y軸變數 (可不選)", options=right_y_axis_options, index=right_y_axis_options.index('Power-Package Power(Watts)') if 'Power-Package Power(Watts)' in right_y_axis_options else 0)

        st.sidebar.header("座標軸範圍設定")
        use_custom_x = st.sidebar.checkbox("自訂X軸範圍")
        x_min = st.sidebar.number_input("X軸最小值 (秒)", value=df['Elapsed Time (s)'].min(), disabled=not use_custom_x)
        x_max = st.sidebar.number_input("X軸最大值 (秒)", value=df['Elapsed Time (s)'].max(), disabled=not use_custom_x)

        use_custom_y1 = st.sidebar.checkbox(f"自訂 '{left_y_axis}' (左Y軸) 範圍")
        y1_min = st.sidebar.number_input(f"左Y軸最小值", disabled=not use_custom_y1, format="%.2f")
        y1_max = st.sidebar.number_input(f"左Y軸最大值", disabled=not use_custom_y1, format="%.2f")

        y2_limits = None
        if right_y_axis and right_y_axis != 'None':
            use_custom_y2 = st.sidebar.checkbox(f"自訂 '{right_y_axis}' (右Y軸) 範圍")
            y2_min = st.sidebar.number_input(f"右Y軸最小值", disabled=not use_custom_y2, format="%.2f")
            y2_max = st.sidebar.number_input(f"右Y軸最大值", disabled=not use_custom_y2, format="%.2f")
            y2_limits = (y2_min, y2_max) if use_custom_y2 else None
        
        st.header("動態比較圖表")
        fig = generate_flexible_chart(df, left_y_axis, right_y_axis, (x_min, x_max) if use_custom_x else None, {'left': (y1_min, y1_max) if use_custom_y1 else None, 'right': y2_limits})
        
        if fig: st.pyplot(fig)
else:
    st.sidebar.info("請上傳您的 Log File 開始分析。")
