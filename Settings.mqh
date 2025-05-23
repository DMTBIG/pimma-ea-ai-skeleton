//+------------------------------------------------------------------+
//|                                                   Settings.mqh  |
//|                                  Copyright 2025, MetaQuotes Ltd. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com"

// URL ของ Flask API
string PythonServiceURL = "http://127.0.0.1:5000";

//+------------------------------------------------------------------+
//|                     Settings.mqh                                  |
//+------------------------------------------------------------------+

//🔧 ====== General Settings ======
input string Sep_General              = "====== General Settings ======";
input double FixedLot                 = 0.01;    // ล็อตคงที่
input double LotSize                  = 0.01;    // ขนาดล็อตหลัก
input int    StopLoss                 = 3000;    // SL พื้นฐาน (points = 300 pips)
input int    TakeProfit               = 5000;    // TP พื้นฐาน (points = 500 pips)
input int    TrailingStop             = 1000;    // Trailing SL (points = 100 pips)
input bool   ManualPause              = false;   // หยุดเทรดด้วยมือ
input string Sep_General_End          = "---------------------------------------------------";

//📊 ====== Indicator Periods ======
input string Sep_Indicators           = "====== Indicator Periods ======";
input int    EMA_Fast_Period          = 21;      // EMA Fast period
input int    EMA_Slow_Period          = 89;      // EMA Slow period
input int    ATR_SL_Period            = 14;      // ATR สำหรับ SL
input int    ATR_Vol_FilterPeriod     = 14;      // ATR สำหรับ Volatility filter
input string Sep_Indicators_End       = "---------------------------------------------------";

//🔐 ====== Static SL/TP Fallback ======
input string Sep_StaticSLTP           = "====== Static SL/TP Fallback ======";
input int    StaticStopLossPoints     = 2000;    // SL เมื่อ AI ไม่พร้อม (points)
input int    StaticTakeProfitPoints   = 3000;    // TP เมื่อ AI ไม่พร้อม (points)
input int    StaticBreakEvenThreshold = 50;      // เบรกอีเว่นเมื่อกำไร ≥ 5 pips
input int    StaticBreakEvenOffset    = 20;      // buffer หลังเบรกอีเว่น (points)
input string Sep_StaticSLTP_End       = "---------------------------------------------------";

//📊 ====== Dynamic SL/TP Settings ======
input string Sep_SLTP_Dynamic         = "====== Dynamic SL/TP Settings ======";
input double SL_TrailingStep          = 2500;    // Trailing Stop (points)
input double SL_MinDistance           = 5000;    // SL ขั้นต่ำ (points)
input double SL_MaxDistance           = 20000;   // SL สูงสุด (points)
input double TPPartialTriggerRatio    = 0.75;    // สัดส่วนถึงจะขยับ TP
input double TPPartialCloseRatio      = 0.50;    // สัดส่วนล็อตที่ปิดใน Partial Exit
input double TPAdjustTriggerRatio     = 0.80;    // เกิน 80% TP → ขยาย TP
input double TPAdjustExtendRatio      = 0.50;    // ขยาย TP เพิ่ม 50%
input string Sep_SLTP_Dynamic_End     = "---------------------------------------------------";

//🔐 ====== Protect Profit (Partial Exit) ======
input string Sep_ProtectProfit        = "====== Protect Profit Settings ======";
input double ProfitProtectThresholdUSD = 5.0;    // เริ่มเลื่อน SL เมื่อกำไร > $5
input int    ProfitProtectBufferPoints = 200;    // buffer หลังเริ่มเลื่อน SL (points)
input double Zone1_CloseRatio         = 0.33;    // ปิด 33% ที่ Z1
input double Zone2_CloseRatio         = 0.33;    // ปิด 33% ที่ Z2
input double Zone3_CloseRatio         = 0.34;    // ปิด 34% ที่ Z3
input string Sep_ProtectProfit_End    = "---------------------------------------------------";

//🧠 ====== AI Decision Settings ======
input string Sep_AI_Decision          = "====== AI Decision Settings ======";
input bool   EnableAIFeatures         = true;    // เปิดใช้งาน AI
input bool   Use_AI_Decision          = true;    // ใช้ AI ตัดสินใจเข้า/ออก
input bool   EnableSelfLearning       = false;   // เปิดการเรียนรู้ด้วยตัวเอง
input bool   EnableOptimizer          = false;   // เปิด Optimizer ภายใน
input double AI_EntryThresholdHigh    = 0.70;    // เกณฑ์เข้า BUY
input double AI_EntryThresholdLow     = 0.30;    // เกณฑ์เข้า SELL
input double AI_ExitPartialThreshold  = 0.15;    // คะแนนต่ำสุดปิดบางส่วน
input double AI_ExitFullThreshold     = 0.95;    // คะแนนสูงสุดปิดทั้งหมด
input int    AIDelayBars              = 10;      // หน่วงบาร์ก่อน entry
input int    AIDelaySeconds           = 60;      // หน่วงเวลา fallback กรณี iBarShift ไม่ทำงาน
input bool   UseNnForEntrySignal      = true;    // ใช้ NN สำหรับสัญญาณเข้า
input bool   UseNnForSlTp             = true;    // ใช้ AI คำนวณ SL/TP
input bool   EnableModelMonitor       = false;   // ตรวจสุขภาพโมเดล
input bool   EnableExplainableAI      = false;   // ใช้งาน Explainable AI
input bool   EnableRegimeAnalysis     = false;   // Regime Analysis
input string Sep_AI_Decision_End      = "---------------------------------------------------";

//🧪 ====== RSI & Override Settings ======
input string Sep_RSI                  = "====== RSI & Override Settings ======";
input bool   ManualOverrideEntry      = false;   // บังคับเข้าไม้แม้ AI ไม่อนุมัติ
input bool   UseAutoOverrideByRSI     = true;    // ใช้ RSI แทนอัตโนมัติ
input int    RSI_Period               = 14;      // Period ของ RSI
input int    RSI_Override_Period      = 9;       // Period สำหรับ Override
input double RSI_Overbought           = 70.0;    // RSI > 70 overbought
input double RSI_Oversold             = 30.0;    // RSI < 30 oversold
input double RSI_Buy_Threshold        = 25.0;    // RSI ต่ำกว่าจะ Buy
input double RSI_Sell_Threshold       = 75.0;    // RSI สูงกว่าจะ Sell
input string Sep_RSI_End              = "---------------------------------------------------";

//🚦 ====== Session & Time Block Filter ======
input string Sep_Time                 = "====== Time & Session Filter ======";
input bool   EnableSessionFilter      = true;    // เปิดกรองช่วงเวลา
input int    SessionStartHour         = 7;       // เริ่ม Session (ชั่วโมง)
input int    SessionEndHour           = 17;      // สิ้นสุด Session (ชั่วโมง)
input bool   EnableTimeBlock          = true;    // ใช้ BlockStartTime/EndTime
input string BlockStartTime          = "22:00"; // เวลาเริ่มบล็อก (HH:MM)
input string BlockEndTime            = "01:00"; // เวลาเลิกบล็อก (HH:MM)
input string Sep_Time_End             = "---------------------------------------------------";

//📊 ====== Spread Filter ======
input string Sep_Spread               = "====== Spread Filter ======";
input bool   EnableSpreadFilter       = true;    // เปิดกรอง Spread
input int    SpreadThresholdPips      = 20;      // Spread สูงสุด (pips)
input string Sep_Spread_End           = "---------------------------------------------------";

//📊 ====== Open Interest & COT Filter ======
input string Sep_OI_Corr              = "====== OI & Correlation Filter ======";
input bool   EnableOpenInterestFilter = true;   // เปิดกรอง OI
input double OpenInterest_Minimum     = 100000; // OI ขั้นต่ำ
input bool   EnableCOTFilter          = true;   // เปิดกรอง COT
input double COT_Threshold            = 75.0;   // COT Threshold (%)
input bool   EnableCorrelationFilter  = true;   // เปิดกรอง Corr
input double CorrelationThreshold     = 0.30;   // Corr Threshold (0–1)
input string Sep_OI_Corr_End          = "---------------------------------------------------";

//📡 ====== News Filter ======
input string Sep_News                 = "====== News Filter ======";
input bool   EnableNewsFilter         = true;   // เปิดกรองข่าว
input bool   BlockOnHighImpactNews    = true;   // บล็อกข่าวแรง
input string Sep_News_End             = "---------------------------------------------------";

//⚙️ ====== StopTrading Logic ======
input string Sep_StopTrading          = "====== StopTrading Logic ======";
input bool   EnableTrading            = true;    // เปิด/ปิดการเทรดทั้งหมด
input int    MaxConsecutiveLosses     = 3;       // ขาดทุนติดต่อกันก่อนหยุด
input double MaxCumulativeLoss        = -500.0;  // ขาดทุนสะสมก่อนหยุด (USD)
input int    AILowScoreThreshold      = 7;       // AI คะแนนต่ำก่อนหยุด
input int    MaxDrawdownExit          = 800;     // Drawdown สูงสุด (USD)
input int    MaxDrawdownPips          = 3000;    // Drawdown สูงสุด (pips)
input int    MaxBarsHolding           = 120;     // แท่งเทียนสูงสุดที่ถือ order
input string Sep_StopTrading_End      = "---------------------------------------------------";

//📡 ====== Server & External Services ======
input string Sep_Services             = "====== Server & External Services ======";
//input string PythonServiceURL         = "http://127.0.0.1:5000"; // URL Python API
input string TradingEconomicsApiKey   = "";       // API Key TradingEconomics
input string GeminiSentimentApiKey    = "";       // API Key Gemini Sentiment
input string Sep_Services_End         = "---------------------------------------------------";

//📝 ====== Display & HUD ======
input string Sep_Display              = "====== Display & AI HUD ======";
input bool   EnableAIDebugHUD         = true;    // แสดง HUD บนกราฟ
input bool   EnableVisualBacktest      = false;   // โหมด Visual Backtest
input double AIDebugScoreThresholdHigh = 0.70;   // สีเขียวเมื่อ ≥
input double AIDebugScoreThresholdLow  = 0.30;   // สีแดงเมื่อ ≤
input int    SmartEntryMinSeconds     = 60;      // รออย่างน้อยกี่วินาทีก่อน entry
input int    SmartEntryFailCooldown   = 300;     // Cooldown เมื่อ fail
input string Sep_Display_End          = "---------------------------------------------------";

//--- Countdown Bar Settings -----------------------------------------
input int    HUD_CountdownWidth       = 180;     // Countdown bar width (px)
input int    HUD_CountdownHeight      = 6;       // Countdown bar height (px)
input color  HUD_CountdownColor       = clrAqua; // Countdown bar fill color

//--- MiniBars Settings ----------------------------------------------
input int    HUD_MiniBarCount         = 5;       // Number of mini bars
input int    HUD_MiniBarWidth         = 20;      // Width of each mini bar (px)
input int    HUD_MiniBarHeight        = 6;       // Height of each mini bar (px)
input int    HUD_MiniBarSpacing       = 4;       // Spacing between mini bars (px)
input color  HUD_MiniBarColor         = clrYellow; // Color of mini bars

//--- Tooltips Settings ---------------------------------------------
input int    HUD_TooltipOffsetX       = 10;      // X offset for tooltips from panel left
input int    HUD_TooltipOffsetY       = 100;     // Y offset for tooltips from panel top
input int    HUD_TooltipLineHeight    = 12;      // Line height for tooltip text
input color  HUD_TooltipColor         = clrWhite;// Tooltip text color

//👁️ ====== Debug Toggles ======
input string Sep_DebugToggles         = "====== Debug Toggles ======";
input bool   ShowDebugHUD             = true;    // ScoreRing/Countdown/MiniBars
input bool   ShowDebugPanels          = true;    // Spread/ATR/VWAP Panels
input bool   ShowDebugArrows          = true;    // Entry/Exit Arrows
input bool   ShowDebugZones           = true;    // SL/TP Zones
input string Sep_DebugToggles_End     = "---------------------------------------------------";

input bool EnableModelOptimization  = true;  // OptimizeModelParameters @23:00
input double DrawdownTargetPct = 2.0;   // เปอร์เซ็นต์ drawdown ที่ยอมรับได้

// === Feature Flags ===
input bool   UseServerPrediction    = true;    // ถ้า true → เรียก Server, false → ใช้ ONNX
input bool   UseONNXPrediction      = false;   // ถ้า true → ตัดสินใจด้วย ONNX
input bool   EnableHTTPS            = true;    // switch to HTTPS URLs
input bool   EnableAutoRetrain      = true;   // เปิด–ปิด Auto-retrain
input bool EnableAutoReloadModel  = true;
//input bool   EnableRemoteCSVExport  = false;   // เปิด export CSV ขึ้น server
//–– Ensemble & Meta-Learner ––
 input bool   UseEnsemblePrediction   = true;   // เรียก Ensemble stack ก่อน
 input bool   UseMetaLearner          = false;  // เรียก Meta-Learner ถัดจาก Ensemble

// === Maintenance Window ===
// EA จะหยุดเรียก WebRequest / Retrain ทุกอย่างในช่วงนี้
input string MaintenanceStart      = "02:00";  // HH:MM (เซิร์ฟเวอร์โซน)
input string MaintenanceEnd        = "02:15";

// === Kill-Switch / Rollback ===
// EA จะเช็ค endpoint นี้เป็นครั้งคราว ถ้า JSON { "panic":true } → ปิดฟีเจอร์ใหม่ทั้งหมด
input bool   EnableKillSwitch      = true;
input string KillSwitchURL         = "https://your-server.com/ea/kill_switch.json";

// ─── Drift Detection & Auto-Retrain ───────────────────────────────
input bool   EnableDriftDetection = true;   // เปิด/ปิดตรวจจับ drift
input double DriftThreshold       = 0.10;   // ถ้า drift ≥ ค่านี้ → สั่ง retrain
input int    RetrainIntervalDays  = 1;      // กำหนด retrain ตามจำนวนวัน (fallback)

//+------------------------------------------------------------------+
//|             Missing Inputs                                       |
//+------------------------------------------------------------------+

//🔍 ====== Filters ======
input bool   UseTrendFilter           = true;    // รัน CheckTrend()
input double MinTrendSlope            = 0.1;     // ค่าชันต่ำสุด
input bool   UseVolatilityFilter      = true;    // รัน CheckVolatility()
input double MinVolatilityATR         = 100.0;   // ATR ขั้นต่ำ
input double ATR_VolatilityThreshold  = 50.0;    // ATR threshold

//🛡️ ====== SL-TP Features ======
input double ATR_SL_Multiplier        = 1.5;     // ใช้ใน ATR based SL
input double SL_Buffer_Pips           = 100.0;   // buffer SL (pips)
input bool   EnableClusterProtection  = true;    // ป้องกัน SL กระจุกกัน
input double SL_ClusterOffset         = 100.0;   // offset (points)
input bool   EnableProtectProfitStatic= true;    // อนุญาต Static SL/TP

//⏱️ ====== Break Even ======
input bool   EnableBreakEven          = true;    // เปิด BE
input double BreakEvenMinProfitPips   = 200;     // เบรกอีเว่นเมื่อ ≥ 200 pips
input double BreakEvenOffsetPips      = 2;       // buffer หลัง BE

//📈 ====== Recovery & Martingale ======
input bool   EnableRecoveryTrade      = true;    // เปิด Recovery
input int    MaxRecoveryTrades        = 2;       // ครั้งสูงสุด
input double RecoveryTriggerATRMult   = 1.5;     // trigger x ATR
input double RecoveryLotPercent       = 0.5;     // lot % ใน Recovery
input int    MartingaleMaxLevel       = 3;       // ระดับสูงสุด

//📉 ====== Risk & StopTrading ======
input double RiskPercentPerTrade      = 1.0;     // % เสี่ยงต่อไม้

// ====== Display & HUD ======
input int    HUD_XOffset      = 20;    // ตำแหน่ง panel จากซ้าย
input int    HUD_YOffset      = 20;    // ตำแหน่ง panel จากบน
input int    HUD_PanelWidth   = 220;   // ความกว้าง panel
input int    HUD_PanelHeight  = 120;   // ความสูง panel
input int    HUD_FontSize     = 12;    // ขนาดฟอนต์
input int    HUD_LineSpacing  = 4;     // ระยะห่างบรรทัด
input string HUD_FontName     = "Courier New";input int    HUD_ZoneOffsetY   = 30;    // เลื่อน Z1–Z3 ลงมา 30px

// ==== Toggle‑Button Settings ====
input int    ToggleButton_OffsetY   = 4;     // ระยะจาก HUD หลัก ลงมา
input int    ToggleButton_Width     = 80;    // ความกว้างปุ่ม (px)
input int    ToggleButton_Height    = 20;    // ความสูงปุ่ม (px)
input int    ToggleButton_FontSize  = 10;    // ขนาดฟอนต์บนปุ่ม
input string ToggleButton_FontName  = "Arial";


//📈 ====== SL/TP & Exit Settings ======
input string Sep_SLTP_Exit          = "====== SL/TP & Exit Settings ======";
input double ATR_ExitMultiplier     = 1.10;    // คูณ ATR เพื่อ exit
input int    ExitSafeHoldingMinutes = 15;      // นาทีถือขั้นต่ำก่อน Trailing/BE
input string Sep_SLTP_Exit_End      = "---------------------------------------------------";

//+------------------------------------------------------------------+
//|            Missing Settings                                      |
//+------------------------------------------------------------------+

// MACD periods
input int    MACD_Fast               = 12;      // Fast period for MACD
input int    MACD_Slow               = 26;      // Slow period for MACD
input int    MACD_Signal             = 9;       // Signal period for MACD

// Draw & HUD
input int    HUD_MiniBarGap          = 4;       // Spacing between mini bars (px)

// Trend & TF filter
input bool   EnableHigherTFTrend     = true;    // ใช้ Multi-TF trend filter
input ENUM_TIMEFRAMES HigherTF_Period= PERIOD_H1; // TF สูงกว่า สำหรับ trend

// SL-TP & Exit
input bool   EnableTrailingDynamic   = true;    // เปิด Trailing SL แบบไดนามิก
input double ATR_TrailingMultiplier  = 2.0;     // คูณ ATR สำหรับ Trailing SL

// StopTrading limits
input double MaxAllowedDrawdownPercent = 3.0;   // Drawdown สูงสุด (%) ก่อนหยุด
input int    MaxHoldingBars            = 120;   // แท่งเทียนสูงสุดที่ถือ order

// Partial Exit Zones
input double Zone1Ratio              = 0.33;    // TP Zone1 ratio
input double Zone2Ratio              = 0.66;    // TP Zone2 ratio

// Profit Goals
input double DailyProfitGoal         = 100.0;   // เป้าหมายกำไรต่อวัน (USD)
input double WeeklyProfitGoal        = 500.0;   // เป้าหมายกำไรต่อสัปดาห์ (USD)
input double MonthlyProfitGoal       = 2000.0;  // เป้าหมายกำไรต่อเดือน (USD)

// Candle & Delay
input int    TotalBarsPerCandle      = 60;      // สำหรับคำนวณ % mini-bars

//+------------------------------------------------------------------+
//| ฟังก์ชันช่วยแปลงค่า Settings เป็นราคา / ค่าต่างๆ               |
//+------------------------------------------------------------------+
// แปลง SL (points) → ราคา
double GetStopLossPrice(double entryPrice)
{
   return entryPrice - StopLoss * _Point;
}
// แปลง TP (points) → ราคา
double GetTakeProfitPrice(double entryPrice)
{
   return entryPrice + TakeProfit * _Point;
}
// แปลง Static SL → ราคา
double GetStaticSLPrice(double entryPrice)
{
   return entryPrice - StaticStopLossPoints * _Point;
}
// แปลง Static TP → ราคา
double GetStaticTPPrice(double entryPrice)
{
   return entryPrice + StaticTakeProfitPoints * _Point;
}
//+------------------------------------------------------------------+
