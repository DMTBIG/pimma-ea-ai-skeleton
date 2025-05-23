//+------------------------------------------------------------------+
//|                                                SettingsFlags.mqh |
//|                                  Copyright 2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
//#pragma once

// === Feature Flags ===
//input bool   UseServerPrediction    = true;    // ถ้า true → เรียก Server, false → ใช้ ONNX
input bool   UseONNXPrediction      = false;   // ถ้า true → ตัดสินใจด้วย ONNX
input bool   EnableHTTPS            = true;    // switch to HTTPS URLs
input bool   EnableAutoRetrain      = false;   // เปิด–ปิด Auto-retrain
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
