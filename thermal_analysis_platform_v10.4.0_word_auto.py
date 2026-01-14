# =============================================================================
# thermal_analysis_platform_v10.4.0_word_auto.py
# 2026 年度升級版：數據分析 + Word 報告一鍵生成
# =============================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import json
from datetime import datetime, date, timedelta
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from docxtpl import DocxTemplate  # <--- 新增 Word 引擎

# 版本資訊
VERSION = "v10.4.0 - Word Automation Ready"
VERSION_DATE = "2026年1月"

# =============================================================================
# 1. 核心自動化組件 (新增：Word 報告導航員)
# =============================================================================

class WordExporter:
    """自動填表機器人：負責將計算好的 summary 填入 Word 標籤中"""
    
    @staticmethod
    def generate_report(summary_df: pd.DataFrame, template_bytes: io.BytesIO) -> io.BytesIO:
        """
        自動對標邏輯：
        將 summary_df 中的 'Ch.' 欄位轉換為 Word 標籤。
        例如：Ch. 1 的數據會對應到 Word 裡的 {{ch1_c1}}
        """
        # 載入範本
        doc = DocxTemplate(template_bytes)
        
        # 構建「數據包」 (Context)
        context = {
            "report_date": date.today().strftime("%Y-%m-%d"),
            "total_channels": len(summary_df)
        }
        
        # 動態對應每個量測點
        for _, row in summary_df.iterrows():
            ch_num = row['Ch.']
            temp_val = row['Result (Case Temp)']
            # 建立對應關係，如 ch1_c1, ch2_c1...
            context[f"ch{ch_num}_c1"] = temp_val
            # 同時也把 Location 傳進去，如果 Word 有需要可以顯示
            context[f"name{ch_num}"] = row['Location']
            
        # 執行填色（渲染）
        doc.render(context)
        
        # 存入記憶體流供下載
        output_stream = io.BytesIO()
        doc.save(output_stream)
        output_stream.seek(0)
        return output_stream

# =============================================================================
# 2. 數據模型與解析系統 (保留原有解析邏輯並優化)
# =============================================================================

@dataclass
class LogMetadata:
    filename: str
    log_type: str
    rows: int
    columns: int
    time_range: str
    file_size_kb: float

class LogData:
    def __init__(self, df: pd.DataFrame, metadata: LogMetadata):
        self.df = df
        self.metadata = metadata
        self._numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    def filter_by_time(self, x_limits: Tuple[float, float]):
        if x_limits is None: return self.df
        x_min_td = pd.to_timedelta(x_limits[0], unit='s')
        x_max_td = pd.to_timedelta(x_limits[1], unit='s')
        return self.df[(self.df.index >= x_min_td) & (self.df.index <= x_max_td)]

# (其餘解析器 Parser 邏輯與 v10.3.8 相同，為節省篇幅在此簡化呈現，
# 但保留完整的 SummaryRenderer 與 UI 工廠邏輯)

# ... [此處省略原有 ParserRegistry, GPUMonParser, PTATParser 實作細節，與你提供的 v10.3.8 一致] ...

# =============================================================================
# 3. UI 呈現層 (新增：Word 下載按鈕)
# =============================================================================

class SummaryRenderer:
    def __init__(self, log_data_list: List[LogData]):
        self.log_data_list = log_data_list

    def render(self):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white;">
            <h3>📋 溫度整合摘要 & Word 報告產出</h3>
            <p>已整合數據，並自動準備好 Word 標籤：{{ch1_c1}} ~ {{ch31_c1}}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 1. 生成摘要表格
        # [此處調用 TemperatureSummaryGenerator.generate_summary_table]
        from __main__ import TemperatureSummaryGenerator # 確保引用到
        summary_df = TemperatureSummaryGenerator.generate_summary_table(self.log_data_list)
        
        if summary_df.empty:
            st.warning("請先上傳檔案以產出摘要。")
            return

        # 2. 預覽表格
        st.markdown("### 🔍 數據預覽")
        st.dataframe(summary_df, use_container_width=True)

        # 3. Word 自動化按鈕 (核心新增)
        st.markdown("---")
        st.markdown("### 📥 產出正式測試報告")
        st.info("💡 請確保您的 Word 範本中已埋入 {{ch1_c1}} 等標籤。")
        
        uploaded_template = st.file_uploader("📂 上傳您的 Word 範本 (.docx)", type=['docx'])
        
        if uploaded_template and st.button("🚀 生成報告並下載"):
            with st.spinner("正在將數據填入 Word 範本..."):
                try:
                    template_bytes = io.BytesIO(uploaded_template.read())
                    report_stream = WordExporter.generate_report(summary_df, template_bytes)
                    
                    st.download_button(
                        label="⬇️ 點擊下載產出的 Word 報告",
                        data=report_stream,
                        file_name=f"Thermal_Test_Report_{date.today()}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                    st.success("✅ 報告生成成功！數據已根據 Ch. 序號自動對應。")
                except Exception as e:
                    st.error(f"❌ 生成失敗：{e}")

# =============================================================================
# 4. 統計與輔助功能 (保留原有 Generator 邏輯)
# =============================================================================

class TemperatureSummaryGenerator:
    @staticmethod
    def generate_summary_table(log_data_list: List[LogData]) -> pd.DataFrame:
        summary_data = []
        ch_num = 1
        for log in log_data_list:
            df = log.df
            numeric_cols = df.select_dtypes(include=['number']).columns
            temp_cols = [c for c in numeric_cols if c not in ['Date', 'sec', 'RT', 'TIME']]
            
            for col in temp_cols:
                max_v = df[col].max()
                clean_name = col.replace('YOKO: ', '').replace('PTAT: ', '').replace('GPU: ', '')
                summary_data.append({
                    'Ch.': ch_num,
                    'Location': clean_name,
                    'Result (Case Temp)': round(max_v, 1) if pd.notna(max_v) else "N/A"
                })
                ch_num += 1
        return pd.DataFrame(summary_data)

# =============================================================================
# 5. 啟動入口 (Main)
# =============================================================================

def main():
    # ... [原有頁面設定與側邊欄邏輯] ...
    # 這裡會觸發各個 Renderer 的渲染功能
    pass

if __name__ == "__main__":
    # 執行主程式邏輯
    # (此處為示意，建議將此完整 Code 與你原有的 v10.3.8 結構合併)
    st.title(f"🚀 {VERSION}")
    # ... 原有主流程 ...
