# final_tool_for_deployment.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- 數據處理函式 ---
# 這個函式接收的是上傳的檔案物件(uploaded_file)，而不是一個固定的檔名
def load_and_clean_data(uploaded_file, header_row_number):
    if uploaded_file is None:
        return None
    try:
        # 關鍵：使用傳入的uploaded_file物件來讀取，而不是寫死的檔名
        df = pd.read_csv(uploaded_file, header=header_row_number - 1, thousands=',', low_memory=False)
        
        df.columns = df.columns.str.strip()
        time_column = 'Time'
        if time_column not in df.columns:
            st.error(f"錯誤：在第 {header_row_number} 行中找不到名為 '{time_column}' 的時間欄位。")
            st.write("偵測到的欄位有：", df.columns.tolist())
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

# --- 圖表繪製與表格生成的函式 (此處省略以節省篇幅，內容與前次相同) ---
def generate_temp_power_chart(df):
    df_filtered = df[df['Elapsed Time (s)'] <= 7200].copy()
    power_col = next((c for c in df.columns if 'Package' in c and 'Watt' in c), None)
    temp_col = next((c for c in df.columns if 'MSR Package Temperature' in c), None)
    if not (power_col and temp_col): return None
    df_filtered.loc[:, 'power_numeric'] = pd.to_numeric(df_filtered[power_col], errors='coerce')
    df_filtered.loc[:, 'temp_numeric'] = pd.to_numeric(df_filtered[temp_col], errors='coerce')
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df_filtered['Elapsed Time (s)'], df_filtered['temp_numeric'], color='tab:red', label='Temperature')
    ax1.axhline(y=100, color='darkviolet', linestyle='--', linewidth=2, label='Tjmax (100°C)')
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', color='tab:red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5); ax1.set_ylim(bottom=40, top=120)
    ax2 = ax1.twinx(); ax2.plot(df_filtered['Elapsed Time (s)'], df_filtered['power_numeric'], color='tab:blue', label='Power')
    ax2.set_ylabel('Power (W)', color='tab:blue', fontsize=12); ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xlim(0, 7200); plt.title('Temperature vs. Power Curve', fontsize=16)
    lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left'); fig.tight_layout()
    return fig

def generate_freq_power_chart(df):
    df_filtered = df[df['Elapsed Time (s)'] <= 7200].copy()
    power_col = next((c for c in df.columns if 'Package' in c and 'Watt' in c), None)
    freq_col = 'CPU0-Frequency(MHz)'
    if not (power_col and freq_col and freq_col in df.columns): return None
    df_filtered.loc[:, 'power_numeric'] = pd.to_numeric(df_filtered[power_col], errors='coerce')
    df_filtered.loc[:, 'freq_numeric'] = pd.to_numeric(df_filtered[freq_col], errors='coerce')
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df_filtered['Elapsed Time (s)'], df_filtered['freq_numeric'], color='tab:green', label='Frequency')
    ax1.axhline(y=1800, color='blue', linestyle='--', linewidth=2, label='HFM (Base Freq)')
    ax1.axhline(y=400, color='orange', linestyle='--', linewidth=2, label='LFM (Min Freq)')
    ax1.set_xlabel('Time (seconds)', fontsize=12); ax1.set_ylabel('Frequency (MHz)', color='tab:green', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:green'); ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2 = ax1.twinx(); ax2.plot(df_filtered['Elapsed Time (s)'], df_filtered['power_numeric'], color='tab:red', label='Power')
    ax2.set_ylabel('Power (W)', color='tab:red', fontsize=12); ax2.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_xlim(0, 7200); plt.title('Frequency vs. Power Curve', fontsize=16)
    lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right'); fig.tight_layout()
    return fig

def generate_stats_df(df):
    df_filtered = df[df['Elapsed Time (s)'] <= 7200].copy()
    temp_col = next((c for c in df.columns if 'MSR Package Temperature' in c), None)
    power_col = next((c for c in df.columns if 'Package' in c and 'Watt' in c), None)
    freq_col = 'CPU0-Frequency(MHz)'
    max_temp = f"{pd.to_numeric(df_filtered[temp_col], errors='coerce').max():.2f}" if temp_col else "N/A"
    min_temp = f"{pd.to_numeric(df_filtered[temp_col], errors='coerce').min():.2f}" if temp_col else "N/A"
    avg_temp = f"{pd.to_numeric(df_filtered[temp_col], errors='coerce').mean():.2f}" if temp_col else "N/A"
    max_power = f"{pd.to_numeric(df_filtered[power_col], errors='coerce').max():.2f}" if power_col else "N/A"
    min_power = f"{pd.to_numeric(df_filtered[power_col], errors='coerce').min():.2f}" if power_col else "N/A"
    avg_power = f"{pd.to_numeric(df_filtered[power_col], errors='coerce').mean():.2f}" if power_col else "N/A"
    max_freq = f"{pd.to_numeric(df_filtered[freq_col], errors='coerce').max():.0f}" if freq_col in df.columns else "N/A"
    min_freq = f"{pd.to_numeric(df_filtered[freq_col], errors='coerce').min():.0f}" if freq_col in df.columns else "N/A"
    avg_freq = f"{pd.to_numeric(df_filtered[freq_col], errors='coerce').mean():.0f}" if freq_col in df.columns else "N/A"
    stats_data = {'Name': ['CPU Package Temp (°C)', 'CPU Package Power (W)', 'CPU0 Freq (MHz)'],'Max': [max_temp, max_power, max_freq],'Min': [min_temp, min_power, min_freq],'Avg': [avg_temp, avg_power, avg_freq]}
    return pd.DataFrame(stats_data)

# --- Streamlit 網頁應用程式介面 ---
st.set_page_config(layout="wide")
st.title("互動式熱功耗數據探索平台")

st.sidebar.header("控制面板")
# 這裡建立檔案上傳元件
uploaded_log_file = st.sidebar.file_uploader("1. 上傳 Log File (.csv)", type="csv")

if uploaded_log_file is not None:
    # 這裡將上傳的檔案物件傳給函式
    master_df = load_and_clean_data(uploaded_log_file, header_row_number=3) 
    
    if master_df is not None:
        st.success("Log File 載入並分析成功！")
        numeric_columns = master_df.select_dtypes(include=['number']).columns.tolist()
        if 'Elapsed Time (s)' in numeric_columns:
            numeric_columns.remove('Elapsed Time (s)')
        
        # UI Elements
        st.sidebar.header("圖表設定")
        left_y_axis = st.sidebar.selectbox("選擇左側Y軸變數", options=numeric_columns, index=numeric_columns.index('Miscellaneous-MSR Package Temperature(Degree C)') if 'Miscellaneous-MSR Package Temperature(Degree C)' in numeric_columns else 0)
        right_y_axis_options = [None] + numeric_columns
        right_y_axis = st.sidebar.selectbox("選擇右側Y軸變數 (可不選)", options=right_y_axis_options, index=right_y_axis_options.index('Power-Package Power(Watts)') if 'Power-Package Power(Watts)' in right_y_axis_options else 0)
        
        # Display Area
        st.header("動態比較圖表")
        # For simplicity, we'll just show the two main charts for now. The flexible chart code can be added back if needed.
        st.pyplot(generate_temp_power_chart(master_df))
        st.pyplot(generate_freq_power_chart(master_df))
        st.header("數據量化呈現")
        st.table(generate_stats_df(master_df))
else:
    st.info("請從左側側邊欄上傳您的 Log File 開始分析。")