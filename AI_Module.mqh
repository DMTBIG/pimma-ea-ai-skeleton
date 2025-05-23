//+------------------------------------------------------------------+
//|                                                    AI_Module.mqh |
//|                                  Copyright 2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#ifndef __AI_MODULE_MQH__
#define __AI_MODULE_MQH__

#include <Trade\Trade.mqh>
#include "Settings.mqh"
#include "FeatureEnrichment.mqh"

extern int ATR_Handle;          // ← เพิ่มบรรทัดนี้: handle ของ ATR หลัก
extern int ATR_Handle_Vol;      // ← เพิ่มบรรทัดนี้: handle ของ ATR สำหรับ Volatility filter

// ─── Ensemble & Meta-Learner APIs ────────────────────────────────

//+------------------------------------------------------------------+
//| AI Module for PIMMA EA                                           |
//| - BuildFeatureString                                            |
//| - SendAIPredictionRequest                                       |
//| - AIDecision handler                                             |
//+------------------------------------------------------------------+

// --- External Inputs จาก EA หลัก ---
// *** ต้องตรวจสอบว่าตัวแปรเหล่านี้ถูกประกาศใน EA หลัก ***
// *** และฟังก์ชันนี้ต้องถูกเรียกโดยส่งค่าเหล่านี้เข้ามา ***
// extern double LotSize; // ไม่ต้องใช้ extern แล้ว
// extern int    StopLoss;
// extern int    TakeProfit;
// extern int    TrailingStop;

// --- ฟังก์ชัน StringTrim (ถ้ายังไม่มี) ---
string StringTrim(string s)
{
    StringTrimLeft(s);
    StringTrimRight(s);
    return s;
}

// === Global variables ===
// extern CTrade trade;  // << ลบบรรทัดนี้ออก ให้ใช้ trade จากไฟล์หลัก
int emaFastHandle;
int emaSlowHandle;
int rsiHandle;
int macdHandle;
// ATR_Handle declared in main EA file
// MagicNumber declared in main EA file, will be passed as parameter where needed
//+------------------------------------------------------------------+
//| Initialize AI Module (call in OnInit)                           |
//+------------------------------------------------------------------+
bool InitializeAI()
{
   emaFastHandle = iMA(_Symbol, _Period, 5,   0, MODE_EMA, PRICE_CLOSE);
   emaSlowHandle = iMA(_Symbol, _Period, 20,  0, MODE_EMA, PRICE_CLOSE);
   rsiHandle     = iRSI(_Symbol, _Period, 14, PRICE_CLOSE);
   macdHandle    = iMACD(_Symbol, _Period, 12, 26, 9, PRICE_CLOSE);

   if(emaFastHandle == INVALID_HANDLE || emaSlowHandle == INVALID_HANDLE ||
      rsiHandle     == INVALID_HANDLE || macdHandle    == INVALID_HANDLE)
   {
      Print("AI Module Init Error: Failed to create indicator handles");
      return(false);
   }
   Print("AI Module: Indicators initialized.");
   return(true);
}

//+------------------------------------------------------------------+
//| Build 40-feature CSV string                                      |
//+------------------------------------------------------------------+
string BuildFeatureString()
{
   // 1) เตรียมอาร์เรย์เก็บ 40 ฟีเจอร์
   double features[45];
   ArrayInitialize(features, 0.0);

   // 2) ดึงค่า EMA Fast, EMA Slow, RSI, MACD Main/Signal 5 ค่า
   double buf1[5], buf2[5], buf3[5], buf4[5], buf5[5];
   if(CopyBuffer(emaFastHandle, 0, 0, 5, buf1) < 5 ||
      CopyBuffer(emaSlowHandle, 0, 0, 5, buf2) < 5 ||
      CopyBuffer(rsiHandle,     0, 0, 5, buf3) < 5 ||
      CopyBuffer(macdHandle,    0, 0, 5, buf4) < 5 ||  // MACD main
      CopyBuffer(macdHandle,    1, 0, 5, buf5) < 5)    // MACD signal
   {
      Print("BuildFeatureString Error: ดึง indicator ไม่ครบ"); 
      return("");
   }
   for(int i = 0; i < 5; i++)
   {
      features[      i] = buf1[i];      // EMA Fast
      features[ 5 + i] = buf2[i];      // EMA Slow
      features[10 + i] = buf3[i];      // RSI
      features[15 + i] = buf4[i];      // MACD Main
      features[20 + i] = buf5[i];      // MACD Signal
   }

   // 3) ดึง ATR หลัก 5 ค่า จาก handle ATR_Handle
   double atrBuf[5];
   if(CopyBuffer(ATR_Handle, 0, 0, 5, atrBuf) < 5)
   {
      Print("BuildFeatureString Error: ดึง ATR หลักไม่ครบ");
      return("");
   }
   for(int j = 0; j < 5; j++)
      features[25 + j] = atrBuf[j];

   // 4) ดึง ATR Volatility 5 ค่า จาก handle ATR_Handle_Vol
   double volAtr[5];
   if(CopyBuffer(ATR_Handle_Vol, 0, 0, 5, volAtr) < 5)
   {
      Print("BuildFeatureString Error: ดึง Volatility ATR ไม่ครบ");
      return("");
   }
   for(int k = 0; k < 5; k++)
      features[30 + k] = volAtr[k];

   // 5) ฟีเจอร์เดี่ยวอีก 5 ค่า
   //    35) Tick Volume   36) Spread (หน่วยจุด)
   //    37–38) ชั่วโมง/นาที (normalize 0–1)  39) Bias term =1.0
   features[35] = (double)iVolume(_Symbol, _Period, 0);
   features[36] = (SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                 - SymbolInfoDouble(_Symbol, SYMBOL_BID)) / _Point;

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   features[37] = (double)dt.hour  / 23.0;
   features[38] = (double)dt.min   / 59.0;

   features[39] = 1.0;  // Bias term

   // --- Feature Enrichment ---
   features[40] = GetGoogleTrends(_Symbol);
   features[41] = GetSocialSentiment("reddit");
   features[42] = GetSocialSentiment("twitter");
   features[43] = GetMacroIndicator("PMI");
   features[44] = GetOrderBookImbalance(_Symbol);

   // 6) ประกอบเป็นสตริง CSV ความละเอียด 6 ตำแหน่ง
   string csv = "";
   int total = ArraySize(features);
   for(int idx = 0; idx < total; idx++)
   {
      csv += DoubleToString(features[idx], 6);
      if(idx < total - 1) 
         csv += ",";
   }
   return(csv);
}

//+------------------------------------------------------------------+
//| Send AI Prediction Request (POST /predict)                       |
//+------------------------------------------------------------------+
bool SendAIPredictionRequest(const string data, double &prediction)
{
   ResetLastError();
   double score = 0.0;

   if(UseEnsemblePrediction)  // ← ถ้าเปิดใช้งาน Ensemble
   {
      if(!SendEnsembleRequest(data, score)) return(false);
   }
   else if(UseMetaLearner)     // ← ถ้าเปิดใช้งาน Meta-Learner
   {
      if(!SendMetaLearnerRequest(data, score)) return(false);
   }
   else                        // fallback: ส่งไป /predict ตามโค้ดเดิม
      {
         // เตรียม endpoint + payload
         string url     = PythonServiceURL + "/predict";
         string hdr     = "Content-Type: application/json\r\n";
         string body    = "{\"command\":\"PREDICT_ENTRY\",\"data\":\"" + data + "\"}";
   
         // แปลงเป็น uchar[]
         uchar tmp[];        int blen = StringToCharArray(body, tmp, 0, WHOLE_ARRAY, CP_UTF8);
         uchar inBuf[];      ArrayResize(inBuf, blen);
         for(int i=0; i<blen; i++) inBuf[i] = (uchar)tmp[i];
         uchar outBuf[];     ArrayResize(outBuf,0);
         string respHdr     = "";
   
         // เรียก WebRequest
         int res = WebRequest("POST", url, hdr, 5000, inBuf, outBuf, respHdr);
         if(res == 200)
         {
            string resp = CharArrayToString(outBuf, 0, WHOLE_ARRAY, CP_UTF8);
            score = StringToDouble(resp);
         }
         else
         {
            PrintFormat("❌ WebRequest /predict failed: %d", res);
            score = -1.0;
         }
      }
       // คืนค่า final
       prediction = score;
       PrintFormat("🎯 Final AI Score: %.4f", prediction);
       return(true);
 }

//+------------------------------------------------------------------+
//| AI Decision - Uses SendAIPredictionRequest for Entry Signal      |
//+------------------------------------------------------------------+
void AIDecision() // This function seems informational, the main logic is in OnTick
{
   if(!Use_AI_Decision) return;

   string feat = BuildFeatureString();
   if(StringLen(feat) == 0) return; // Don't proceed if features failed

   double score = 0.0;
   // --- แก้ไข: เรียก SendAIPredictionRequest โดยไม่มี command ---
   if(!SendAIPredictionRequest(feat, score)) return;

   // This PrintFormat is likely redundant if called from OnTick which also prints the score
   // PrintFormat("AIDecision Function: AI Score=%.3f", score);
}

//+------------------------------------------------------------------+
//| Parse JSON (Example - not directly used by SendAIPredictionRequest) |
//+------------------------------------------------------------------+
bool ParseJsonResult(const char &buf[], double &score)
{
   string resp = CharArrayToString(buf, 0, WHOLE_ARRAY, CP_UTF8);
   int pos = StringFind(resp, "\"result\":");
   if(pos < 0)
   {
      Print("ParseJsonResult Error: '\"result\":' not found in response: ", resp);
      return(false);
   }
   string val_part = StringSubstr(resp, pos + StringLen("\"result\":"));
   StringTrimLeft(val_part);
   StringTrimRight(val_part);

   // --- แก้ไข: ใช้ StringFind แทน StringGetCharacter ---
   int end_pos_comma = StringFind(val_part, ",", 0);  // ค้นหา comma
   int end_pos_brace = StringFind(val_part, "}", 0);  // ค้นหา brace ปิด
   // --- จบการแก้ไข ---

   int end_pos = -1;
   // หาตำแหน่งที่เจอก่อน (ถ้าเจอทั้งคู่)
   if(end_pos_comma >= 0 && end_pos_brace >= 0)
       end_pos = MathMin(end_pos_comma, end_pos_brace);
   else if (end_pos_comma >= 0)
       end_pos = end_pos_comma;
   else if (end_pos_brace >= 0)
       end_pos = end_pos_brace;
   // ถ้าไม่เจอทั้งคู่ อาจจะเป็นแค่ตัวเลขเดี่ยวๆ

   string val;
   if(end_pos >= 0)
      val = StringSubstr(val_part, 0, end_pos); // ตัดเอาเฉพาะส่วนตัวเลข
   else
      val = val_part; // ใช้ทั้งส่วนที่เหลือ

   score = StringToDouble(val);
   if(GetLastError() != 0)
   {
        Print("ParseJsonResult Error: Failed to convert '", val, "' to double.");
        return false;
   }
   return(true);
}

//+------------------------------------------------------------------+
//| Visual Backtest Insights (Called from OnDeinit)                  |
//+------------------------------------------------------------------+
// --- แก้ไข: เพิ่มพารามิเตอร์ magic_num ---
void VisualizeBacktest(const int magic_num)
{
   if(!EnableVisualBacktest) return;
   if(!MQLInfoInteger(MQL_TESTER)) // Only run in tester
   {
       // Print("VisualizeBacktest: Not running in Tester mode. Skipping visualization.");
       return;
   }

   if(!HistorySelect(0, TimeCurrent()))
   {
       Print("VisualizeBacktest Error: HistorySelect failed.");
       return;
   }
   int deals = HistoryDealsTotal();
   if(deals <= 0)
   {
       Print("VisualizeBacktest: No deals found in history.");
       return;
   }

   double pnl[];
   ArrayResize(pnl, deals); // Initial size, will resize later
   int count = 0;
   for(int i = 0; i < deals; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      // --- แก้ไข: ใช้ magic_num ที่รับมา ---
      if(ticket > 0 && HistoryDealGetInteger(ticket, DEAL_MAGIC) == magic_num)
      {
         if(count >= ArraySize(pnl)) ArrayResize(pnl, count + 50); // Resize if needed
         pnl[count] = HistoryDealGetDouble(ticket, DEAL_PROFIT);
         count++;
      }
   }
   ArrayResize(pnl, count); // Final resize to actual count

   // --- แก้ไข: ใช้ magic_num ใน Print statement ---
   if(count <= 0)
   {
       PrintFormat("VisualizeBacktest: No deals found with MagicNumber %d", magic_num);
       return;
   }
   PrintFormat("VisualizeBacktest: Found %d deals for MagicNumber %d", count, magic_num);

   // 2) Create JSON payload
   string arr = "[";
   for(int i=0; i<count; i++)
      arr += DoubleToString(pnl[i], 2) + (i<count-1 ? "," : "");
   arr += "]";

   string body    = "{\"pnl\":" + arr + "}";
   char   inBuf[];
   ArrayResize(inBuf, StringToCharArray(body, inBuf, CP_UTF8));
   char   outBuf[]; ArrayResize(outBuf, 0);

   string endpoint = "/visualize";
   string url      = PythonServiceURL + endpoint;
   string hdr      = "Content-Type: application/json\r\n";
   string respHdr  = "";
   int    timeout  = 15000;

   // 3) Call /visualize endpoint
   int res = WebRequest("POST", url, hdr, timeout, inBuf, outBuf, respHdr);

   // 4) Handle response
   if(res == 200)
   {
      Print("VisualizeBacktest: Request to ", endpoint, " successful. Response: ", CharArrayToString(outBuf));
   }
   else
   {
      int lastError = GetLastError();
      PrintFormat("VisualizeBacktest Error: Request to %s failed. Code: %d, LastError: %d", url, res, lastError);
   }
}
// 3.1 CheckCOT – ดึงข้อมูล net_long จาก /cot
bool CheckCOT()
{
   if(!EnableCOTFilter) return true; // ✅ ถ้าไม่เปิดใช้งานตัวกรอง COT ให้ผ่านเลยทันที

   uchar inBuf[], outBuf[];
   string hdr = "Content-Type: application/json\r\n", resHdr = "";
   ArrayResize(inBuf, 0);
   ArrayResize(outBuf, 0);

   int res = WebRequest("GET", PythonServiceURL + "/cot", hdr, 5000, inBuf, outBuf, resHdr);
   if(res != 200)
   {
      PrintFormat("CheckCOT Warning: Failed to get COT data from %s. Code: %d, Error: %d. Allowing trade.",
                  PythonServiceURL + "/cot", res, GetLastError());
      return true; // ✅ ถ้าเรียกไม่สำเร็จ ให้ผ่าน (ไม่บล็อกการเทรด)
   }

   string json = CharArrayToString(outBuf, 0, WHOLE_ARRAY, CP_UTF8);
   int pos = StringFind(json, "\"net_long\":");
   if(pos < 0)
   {
      Print("CheckCOT Error: 'net_long' field not found in response.");
      return true; // ✅ ไม่พบข้อมูล → ผ่าน
   }

   double netLong = StringToDouble(StringSubstr(json, pos + 11, 6));
   PrintFormat("COT: net long=%.1f%%", netLong);

   if(netLong >= COT_Threshold)
   {
      PrintFormat("❌ COT filter blocked: %.1f%% >= %.1f%%", netLong, COT_Threshold);
      return false;
   }

   return true;
}


// 3.2 CheckOpenInterest – ตรวจสอบปริมาณ OI จาก /openinterest
bool CheckOpenInterest()
{
   if(!EnableOpenInterestFilter) return true;  // << เพิ่มบรรทัดนี้ไว้ด้านบนสุด

   uchar inBuf[], outBuf[];
   string hdr = "Content-Type: application/json\r\n", resHdr = "";
   ArrayResize(inBuf, 0);
   ArrayResize(outBuf, 0);

   int res = WebRequest("GET", PythonServiceURL + "/openinterest", hdr, 5000, inBuf, outBuf, resHdr);
   if(res != 200)
   {
      PrintFormat("CheckOpenInterest Warning: Failed to get OI data. Code: %d, Error: %d. Allowing trade.", res, GetLastError());
      return true;
   }

   string text = CharArrayToString(outBuf, 0, WHOLE_ARRAY, CP_UTF8);
   double oi = StringToDouble(text);
   PrintFormat("OI: current=%.0f", oi);
   return (oi > OpenInterest_Minimum);
}

//+------------------------------------------------------------------+
//| ส่งผลกำไรไปยังเซิร์ฟเวอร์ AI                                  |
//+------------------------------------------------------------------+
void LogResultToServer(ulong ticket, double result)
{
   // ใช้ WebRequest ส่งไปที่ /log_result
   string body = StringFormat("{\"ticket\":%d,\"result\":%.2f}", ticket, result);
   uchar inBuf[], outBuf[];
   ArrayResize(inBuf, StringToCharArray(body, inBuf, CP_UTF8));
   ArrayResize(outBuf, 0);

   string hdr = "Content-Type: application/json\r\n", resHdr = "";
   int res = WebRequest("POST", PythonServiceURL + "/log_result", hdr, 5000, inBuf, outBuf, resHdr);

   if(res == 200)
      PrintFormat("✅ LogResultToServer: OK for #%d → %.2f", ticket, result);
   else
      PrintFormat("❌ LogResultToServer ERROR #%d → %.2f | res=%d err=%d", ticket, result, res, GetLastError());
}


bool AI_OptimizeConfig(double current_LotSize, int current_StopLoss, int current_TakeProfit, int current_TrailingStop)
{
    string url     = PythonServiceURL + "/optimize";
    string headers = "Content-Type: application/json\r\n";

    // สร้าง JSON โดยใช้ค่า Parameter ที่รับเข้ามา
    string json = StringFormat("{\"LotSize\":%.2f,\"StopLoss\":%d,\"TakeProfit\":%d,\"TrailingStop\":%d}",
                               current_LotSize, current_StopLoss, current_TakeProfit, current_TrailingStop);

    uchar inBuf[], outBuf[];
    // ... (ส่วนจัดการ Buffer และ WebRequest เหมือนเดิม) ...
    int jsonLen = StringToCharArray(json, inBuf, 0, WHOLE_ARRAY, CP_UTF8);
    if (jsonLen > 0 && inBuf[jsonLen - 1] == 0) ArrayResize(inBuf, jsonLen - 1);
    else if (jsonLen <= 0 && StringLen(json) > 0) { Print("Error converting optimizer json to char array."); return false; }
    ArrayResize(outBuf, 0);
    string respHeaders = "";
    int res = WebRequest("POST", url, headers, 10000, inBuf, outBuf, respHeaders);

    if(res == 200)
    {
        string result = CharArrayToString(outBuf, 0, WHOLE_ARRAY, CP_UTF8);
        PrintFormat("✅ AI Optimizer Received Raw Response: %s", result); // แสดงผลดิบ

        // ++++++++++++++ ลบส่วน Parse และ Assign ค่าทิ้งทั้งหมด ++++++++++++++
        /*
        int pos = StringFind(result, "\"optimized\":");
        if(pos >= 0)
        {
            // ... โค้ด Parse JSON ทั้งหมดที่เคยมี ลบทิ้ง ...
            // ... รวมถึง Loop for และการ Assign ค่า LotSize = val; ...
            // ... และ PrintFormat ตอนท้ายที่ Comment Out ไป ...
        } else { Print("AI Optimizer Error: Key 'optimized' not found in response: ", result); }
        */
        // ++++++++++++++ จบการลบ ++++++++++++++

        Print("✅ AI Optimizer request successful. Check the raw response above for suggested parameters.");
        return true; // แจ้งว่า WebRequest สำเร็จ
    }
    else { PrintFormat("❌ AI Optimizer failed: HTTP %d | Error=%d", res, GetLastError()); }
    return false;
}
//+------------------------------------------------------------------+
//| ATR Exit Reversal Logic                                         |
//+------------------------------------------------------------------+
bool ShouldExitByATRReversal()
{
   if(!PositionSelect(_Symbol)) return false;
   double atr = iATR(_Symbol, _Period, 14);
   if(atr <= 0.0) return false;

   MqlRates rates[];
   if(CopyRates(_Symbol, _Period, 0, 2, rates) < 2) return false;
   ArraySetAsSeries(rates, true);

   double move = MathAbs(rates[0].close - rates[1].close);
   return (move > ATR_ExitMultiplier * atr);
}
//+------------------------------------------------------------------+
//| CheckFeatureDrift: POST /drift                                   |
//+------------------------------------------------------------------+
bool CheckFeatureDrift(const string data, double &driftValue)
{
   if(!EnableDriftDetection) return(false);

   // แปลง CSV string → uchar[]
   uchar inBuf[]; ArrayResize(inBuf,0);
   int len = StringToCharArray(data, inBuf, 0, WHOLE_ARRAY, CP_UTF8);
   ArrayResize(inBuf, len);

   uchar outBuf[]; ArrayResize(outBuf,0);
   string respHdr="";

   int res = WebRequest(
      "POST",
      PythonServiceURL + "/drift",
      "Content-Type: application/json\r\n",
      10000,
      inBuf, outBuf, respHdr
   );
   if(res!=200)
   {
      PrintFormat("❌ DriftDetection failed: HTTP %d", res);
      return(false);
   }

   // อ่าน plain-text ผลลัพธ์แล้วแปลงเป็น double
   string txt = CharArrayToString(outBuf,0,WHOLE_ARRAY,CP_UTF8);
   driftValue = StringToDouble(txt);

   return(driftValue >= DriftThreshold);
}

//+------------------------------------------------------------------+
//| SendEnsembleRequest: POST /predict/ensemble                      |
//+------------------------------------------------------------------+
bool SendEnsembleRequest(const string data, double &result)
{
   // เตรียม JSON body
   string body = "{\"features\":[" + data + "]}";

   // แปลงเป็น uchar[]
   uchar inBuf[]; int len = StringToCharArray(body, inBuf, 0, WHOLE_ARRAY, CP_UTF8);
   ArrayResize(inBuf, len);
   uchar outBuf[]; ArrayResize(outBuf, 0);
   string respHdr;

   // WebRequest
   int res = WebRequest("POST",
                        PythonServiceURL + "/predict/ensemble",
                        "Content-Type: application/json\r\n",
                        10000,
                        inBuf, outBuf, respHdr);
   if(res != 200)
   {
      PrintFormat("EnsembleRequest failed: HTTP %d", res);
      return(false);
   }
   result = StringToDouble(CharArrayToString(outBuf, 0, WHOLE_ARRAY, CP_UTF8));
   return(true);
}

//+------------------------------------------------------------------+
//| SendMetaLearnerRequest: POST /predict/meta                      |
//+------------------------------------------------------------------+
bool SendMetaLearnerRequest(const string data, double &result)
{
   // เตรียม JSON body
   string body = "{\"preds\":[" + data + "]}";

   // แปลงเป็น uchar[]
   uchar inBuf[]; int len = StringToCharArray(body, inBuf, 0, WHOLE_ARRAY, CP_UTF8);
   ArrayResize(inBuf, len);
   uchar outBuf[]; ArrayResize(outBuf, 0);
   string respHdr;

   // WebRequest
   int res = WebRequest("POST",
                        PythonServiceURL + "/predict/meta",
                        "Content-Type: application/json\r\n",
                        10000,
                        inBuf, outBuf, respHdr);
   if(res != 200)
   {
      PrintFormat("MetaLearnerRequest failed: HTTP %d", res);
      return(false);
   }
   result = StringToDouble(CharArrayToString(outBuf, 0, WHOLE_ARRAY, CP_UTF8));
   return(true);
}

//+------------------------------------------------------------------+
//| SendRLPredictRequest: POST /rl/predict                           |
//+------------------------------------------------------------------+
bool SendRLPredictRequest(const string data, int &action)
{
   string url = PythonServiceURL + "/rl/predict";
   // แปลง CSV string → JSON body
   string body = "{\"features\":[" + data + "]}";
   // … (เหมือน SendEnsembleRequest) …
   uchar inBuf[]; int len=StringToCharArray(body,inBuf,0,WHOLE_ARRAY,CP_UTF8);
   ArrayResize(inBuf,len);
   uchar outBuf[]; ArrayResize(outBuf,0);
   string hdr="Content-Type:application/json\r\n",respHdr;
   int res = WebRequest("POST",url,hdr,10000,inBuf,outBuf,respHdr);
   if(res!=200) return(false);
   action = (int)StringToInteger(CharArrayToString(outBuf));
   return(true);
}

//+------------------------------------------------------------------+
//| SendRLStoreRequest: POST /rl/store                               |
//+------------------------------------------------------------------+
bool SendRLStoreRequest(const string obs, int action, double reward, const string next_obs, bool done)
{
   string url = PythonServiceURL + "/rl/store";
   // สร้าง JSON body
   string body = "{\"obs\":[" + obs + "],\"action\":" + IntegerToString(action)
               + ",\"reward\":" + DoubleToString(reward,6)
               + ",\"next_obs\":[" + next_obs + "]"
               + ",\"done\":" + (done?"true":"false") + "}";
   // … (เหมือน above) …
   uchar inBuf[]; int len=StringToCharArray(body,inBuf,0,WHOLE_ARRAY,CP_UTF8);
   ArrayResize(inBuf,len);
   uchar outBuf[]; ArrayResize(outBuf,0);
   string hdr="Content-Type:application/json\r\n",respHdr;
   int res=WebRequest("POST",url,hdr,10000,inBuf,outBuf,respHdr);
   return(res==200);
}

//+------------------------------------------------------------------+
//| SendRLUpdateRequest: POST /rl/update                             |
//+------------------------------------------------------------------+
bool SendRLUpdateRequest()
{
   string url = PythonServiceURL + "/rl/update";
   uchar inBuf[]; ArrayResize(inBuf,0);
   uchar outBuf[]; ArrayResize(outBuf,0);
   string hdr="Content-Type:application/json\r\n",respHdr;
   int res=WebRequest("POST",url,hdr,10000,inBuf,outBuf,respHdr);
   return(res==200);
}

//+------------------------------------------------------------------+
//| SendExplainRequest: POST /explain                                |
//+------------------------------------------------------------------+
bool SendExplainRequest(const string data, double &baseValue, double &shapValues[])
{
   // 1) build payload
   string body = "{\"features\":[" + data + "]}";
   uchar  inBuf[]; int len = StringToCharArray(body, inBuf, 0, WHOLE_ARRAY, CP_UTF8);
   ArrayResize(inBuf, len);
   uchar outBuf[]; ArrayResize(outBuf,0);
   string hdr="Content-Type:application/json\r\n", respHdr;

   // 2) call WebRequest
   int res = WebRequest("POST", PythonServiceURL + "/explain", hdr, 10000, inBuf, outBuf, respHdr);
   if(res != 200)
   {
      PrintFormat("❌ ExplainRequest failed: %d", res);
      return(false);
   }

   // 3) parse JSON response
   string json = CharArrayToString(outBuf, 0, WHOLE_ARRAY, CP_UTF8);
   // ดึง base_value
   int p = StringFind(json, "\"base_value\":");
   string tmp = StringSubstr(json, p + 13);
   StringTrimLeft(tmp); StringTrimRight(tmp);
   int end = StringFind(tmp, ",");
   baseValue = StringToDouble(StringSubstr(tmp, 0, end));

   // ดึง shap_values array
   p = StringFind(json, "\"shap_values\":");
   tmp = StringSubstr(json, p + 15);
   StringTrimLeft(tmp); StringTrimRight(tmp);
   int start = StringFind(tmp, "[")+1, finish = StringFind(tmp, "]");
   string list = StringSubstr(tmp, start, finish-start);
   // แปลงเป็น double[]
   // แบ่ง string ด้วย comma
   int count = 0;
   for(int pos=0; pos >= 0; )
   {
      int comma = StringFind(list, ",", pos);
      string num = (comma>0 ? StringSubstr(list, pos, comma-pos) : StringSubstr(list, pos));
      shapValues[count++] = StringToDouble(num);
      pos = comma>0 ? comma+1 : -1;
   }
   return(true);
}

//+------------------------------------------------------------------+
//| GetPositionSize: POST /position_size                            |
//+------------------------------------------------------------------+
bool GetPositionSize(double equity, double volatility, double drawdown_pct, double &lotSize)
{
    string url  = PythonServiceURL + "/position_size";
    string body = StringFormat("{\"equity\":%.2f,\"volatility\":%.4f,\"drawdown_pct\":%.2f}",
                               equity, volatility, drawdown_pct);
    // … (WebRequest แปลง body→inBuf, ส่ง, parse JSON เพื่อดึง lot_size)
    // ตัวอย่าง:
    uchar inBuf[]; int len=StringToCharArray(body,inBuf,0,WHOLE_ARRAY,CP_UTF8);
    ArrayResize(inBuf,len);
    uchar outBuf[]; ArrayResize(outBuf,0);
    string hdr="Content-Type:application/json\r\n", respHdr;
    int res=WebRequest("POST",url,hdr,10000,inBuf,outBuf,respHdr);
    if(res!=200) return(false);
    string json=CharArrayToString(outBuf,0,WHOLE_ARRAY,CP_UTF8);
    // parse {"lot_size":X.XX}
    int p=StringFind(json,"\"lot_size\":");
    string v=StringSubstr(json,p+11);
    lotSize=StringToDouble(v);
    return(true);
}

//+------------------------------------------------------------------+
//| GetScenarioAnalysis: POST /scenario_analysis                    |
//+------------------------------------------------------------------+
bool GetScenarioAnalysis(const string features, string &analysisJson)
{
    string url  = PythonServiceURL + "/scenario_analysis";
    string body = "{\"features\":[" + features + "]}";
    uchar inBuf[]; int len=StringToCharArray(body,inBuf,0,WHOLE_ARRAY,CP_UTF8);
    ArrayResize(inBuf,len);
    uchar outBuf[]; ArrayResize(outBuf,0);
    string hdr="Content-Type:application/json\r\n", respHdr;
    int res=WebRequest("POST",url,hdr,10000,inBuf,outBuf,respHdr);
    if(res!=200) return(false);
    analysisJson = CharArrayToString(outBuf,0,WHOLE_ARRAY,CP_UTF8);
    return(true);
}

#endif // __AI_MODULE_MQH__