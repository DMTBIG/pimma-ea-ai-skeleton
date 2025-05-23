//+------------------------------------------------------------------+
//|                                                    AI_Conditions.mqh |
//|            รวมฟังก์ชันสำหรับตัวกรองภายนอกของ AI Trading System      |
//+------------------------------------------------------------------+

#ifndef __AI_CONDITIONS_MQH__
#define __AI_CONDITIONS_MQH__

int EMA_Fast_Handle, EMA_Slow_Handle;
int ATR_Handle_Vol;
bool skipWebFilters      = false;  // ข้ามเฉพาะ WebRequest-based filters
bool skipStopTrading     = false;  // ข้าม StopTrading logic (ถ้าต้องการ)

#include <Trade\SymbolInfo.mqh> // อาจจะไม่จำเป็น ถ้า SymbolInfo ถูก include ใน AI_Module แล้ว
#include "AI_Module.mqh"         // <<< สำคัญ: เพื่อให้รู้จัก Input และ PythonServiceURL
#include "Settings.mqh"


//+------------------------------------------------------------------+
//| ตรวจสอบ Spread                                                  |
//+------------------------------------------------------------------+
bool CheckSpread()
{
   // << ย้ายโค้ด CheckSpread เดิมจาก PIMMA_EA_AI_Skeleton.mq5 มาใส่ที่นี่ >>
   // << แก้ไขให้ใช้ SpreadThresholdPips ที่ประกาศ extern ด้านบน >>
   if(!EnableSpreadFilter) return true; // ตัวอย่างการเช็ค Flag (ถ้าต้องการ)

   long currentSpreadPoints = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   if(currentSpreadPoints < 0) {
       Print("CheckSpread Warning: Could not retrieve spread.");
       return true; // หรือ false ตามต้องการ
   }
   // ใช้ SpreadThresholdPips ที่เป็น int (หน่วย point) โดยตรง
   if(currentSpreadPoints > SpreadThresholdPips)
   {
      PrintFormat("❌ CheckSpread Filter: Spread %d points > Threshold %d points", currentSpreadPoints, SpreadThresholdPips);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| ตรวจสอบ Session Time                                           |
//+------------------------------------------------------------------+
bool CheckSession()
{
   // << ย้ายโค้ด CheckSession เดิมจาก PIMMA_EA_AI_Skeleton.mq5 มาใส่ที่นี่ >>
   // << ใช้ SessionStartHour และ SessionEndHour ที่ประกาศ extern ด้านบน >>
   if(!EnableSessionFilter) return true; // ตัวอย่างการเช็ค Flag

   MqlDateTime t;
   TimeToStruct(TimeCurrent(),t); // อาจจะใช้ TimeCurrent() หรือ TimeLocal() แล้วแต่ความต้องการ
   int h=t.hour;

   // แก้ไขเงื่อนไข <= เป็น < ถ้าไม่ต้องการรวมชั่วโมงสุดท้าย
   if(h >= SessionStartHour && h < SessionEndHour)
      return true;

   // PrintFormat("❌ CheckSession: Current hour %d is outside session %d-%d", h, SessionStartHour, SessionEndHour); // เอา Log ออกถ้าไม่ต้องการ
   return false;
}

//+------------------------------------------------------------------+
//| กรองข่าว High-Impact + เสริมด้วย Gemini Sentiment             |
//+------------------------------------------------------------------+
bool CheckNewsHighImpact()
{
   // ถ้า skipWebFilters → ข้ามตรวจกรองข่าวทั้งหมด (อนุญาตให้เทรด)
   if(skipWebFilters) 
      return false;
   // ถ้าไม่เปิดระบบกรองข่าว → อนุญาตให้เทรด
   if(!EnableNewsFilter) 
      return false;

   // --- 1) เรียก Python news API ---
   uchar inBuf[], outBuf[];
   ArrayResize(inBuf,  0);
   ArrayResize(outBuf, 0);
   string headers   = "Content-Type: application/json\r\n";
   string resultHdr = "";
   int res = WebRequest("GET",
                        PythonServiceURL + "/news",
                        headers,
                        5000,
                        inBuf, outBuf,
                        resultHdr);
   if(res != 200)
   {
      PrintFormat("❌ NewsFilter: failed res=%d err=%d → Allowing", res, GetLastError());
      return false;  // ดึงข้อมูลไม่ได้ → อนุญาตให้เทรด
   }

   // แปลงผลลัพธ์เป็น string
   string json = CharArrayToString(outBuf, 0, WHOLE_ARRAY, CP_UTF8);

   // --- 2) ถ้ามีข่าวแรง (impact high) ---
   if(StringFind(json, "\"impact\":\"high\"") != -1)
   {
      if(BlockOnHighImpactNews)
      {
         Print("❌ NewsFilter: High-impact news detected.");
         return true;   // บล็อกการเทรด
      }
      else
      {
         Print("⚠️ NewsFilter: High-impact news detected, but NOT blocking due to settings.");
         // แม้เจอข่าวแรง แต่ผู้ใช้ตั้งค่าให้ไม่บล็อก → ไปตรวจ Gemini ต่อ
      }
   }

   // --- 3) เสริมด้วย Gemini Sentiment API ---
   double gemScore;
   if(GetGeminiSentiment(gemScore))
   {
      PrintFormat("🔍 Gemini Sentiment for %s → %.2f", _Symbol, gemScore);
      // ถ้า sentiment ต่ำกว่า 0.30 → บล็อก
      if(gemScore < 0.30)
      {
         Print("❌ Blocked by Gemini Sentiment (too negative)");
         return true;
      }
   }
   else
   {
      // เรียกไม่สำเร็จ → ปล่อยผ่าน
      //Print("ℹ️ GeminiSentiment API call failed or no API key, allowing trade");
   }

   // ไม่มีเหตุให้บล็อก → อนุญาตให้เทรด
   return false;
}


 void AdjustAIScoreByNews(double &score)
 {
    char inBuf[], outBuf[];         // เปลี่ยนเป็น char
    ArrayResize(inBuf, 0);
    ArrayResize(outBuf, 0);

     string headers    = "Content-Type: application/json\r\n";
     string resultHdr  = "";

    int res = WebRequest("GET",
                         PythonServiceURL + "/news",
                         headers,
                         3000,
                         inBuf,       // char[]
                         outBuf,      // char[]
                         resultHdr);  // string&

     if(res != 200)
     {
        PrintFormat("AdjustAIScoreByNews: API fail res=%d err=%d", res, GetLastError());
        return;
     }
     // … parse json, ปรับ score …
 }

//+------------------------------------------------------------------+
//| ตรวจสอบ Correlation                                             |
//+------------------------------------------------------------------+
bool CheckCorrelation()
{
   if(skipWebFilters) return(true);     // <-- ข้าม filter ใน tester
   if(!EnableCorrelationFilter) return(true);

   char sendBuf[], recvBuf[]; ArrayResize(sendBuf,0); ArrayResize(recvBuf,0);
   string respHdr="";
   int res = WebRequest(
      "GET",
      PythonServiceURL + "/correlation?assets=DXY,BOND,OIL,XAU",
      "", 5000,
      sendBuf, recvBuf,
      respHdr
   );
   if(res != 200)
   {
      PrintFormat("CheckCorrelation: WebRequest failed res=%d err=%d → bypass", res, GetLastError());
      return true;
   }
   string js = CharArrayToString(recvBuf,0,WHOLE_ARRAY,CP_UTF8);
   string dxyStr = GetJsonValue(js,"DXY");
   if(StringLen(dxyStr)==0) return true;
   double dxy = StringToDouble(dxyStr);
   PrintFormat("Correlation DXY = %.2f (Threshold %.2f)", dxy, CorrelationThreshold);
   return (dxy < CorrelationThreshold);
}


//+------------------------------------------------------------------+
//| ตรวจสอบสุขภาพของโมเดล AI                                       |
//+------------------------------------------------------------------+
bool CheckModelHealth()
{
   // << ย้ายโค้ด CheckModelHealth เดิมจาก PIMMA_EA_AI_Skeleton.mq5 มาใส่ที่นี่ >>
   // << ใช้ PythonServiceURL และ EnableModelMonitor ที่ประกาศ extern ด้านบน >>
   if(skipWebFilters) return(true);     // <-- ข้าม filter ใน tester
   if(!EnableCorrelationFilter) return(true);
   char inBuf[];  ArrayResize(inBuf,0); char outBuf[]; ArrayResize(outBuf,0);
   string respHdr=""; string endpoint = "/monitor"; string url = PythonServiceURL + endpoint; int timeout = 5000;
   int res=WebRequest("GET", url, "", timeout, inBuf, outBuf, respHdr);
   if(res == 200) {
      string json = CharArrayToString(outBuf, 0, WHOLE_ARRAY, CP_UTF8);
      if(StringLen(json) > 0) {
          // PrintFormat("CheckModelHealth: OK. Response: %s", json); // อาจจะเอา Log ออก
          // TODO: Parse json เพื่อเช็คสถานะละเอียดขึ้น ถ้าต้องการ
          return(true);
      } else {
          PrintFormat("CheckModelHealth Error: Request OK but empty body.");
          return false;
      }
   } else {
      PrintFormat("CheckModelHealth Error: Request failed. Code: %d, Error: %d", res, GetLastError());
      return(false); // ถ้า Monitor ไม่ได้ ควรถือว่าไม่ผ่าน
   }
}

string GetJsonValue(const string json, const string key)
{
   string searchKey = "\"" + key + "\":"; int keyPos = StringFind(json, searchKey);
   if (keyPos < 0) return ""; int valueStart = keyPos + StringLen(searchKey);
   while(valueStart < StringLen(json) && StringGetCharacter(json, valueStart) == ' ') valueStart++;
   int valueEnd = -1;
   if (StringGetCharacter(json, valueStart) == '"') {
      valueStart++; valueEnd = StringFind(json, "\"", valueStart);
   } else {
      int commaPos = StringFind(json, ",", valueStart); int bracePos = StringFind(json, "}", valueStart);
      if (commaPos >= 0 && bracePos >= 0) valueEnd = MathMin(commaPos, bracePos);
      else if (commaPos >= 0) valueEnd = commaPos; else if (bracePos >= 0) valueEnd = bracePos; else valueEnd = StringLen(json);
   }
   if (valueEnd < 0) return ""; return StringSubstr(json, valueStart, valueEnd - valueStart);
}

bool IsTrendConfirmed()
{
   if(!UseTrendFilter) return true;

   double emaFast[3], emaSlow[3];
   if(CopyBuffer(EMA_Fast_Handle, 0, 0, 3, emaFast) != 3 ||
      CopyBuffer(EMA_Slow_Handle, 0, 0, 3, emaSlow) != 3)
   {
      Print("⚠️ Failed to read EMA buffers");
      return false;
   }

   double slopeFast = emaFast[0] - emaFast[2];
   double slopeSlow = emaSlow[0] - emaSlow[2];

   if(slopeFast > MinTrendSlope && emaFast[0] > emaSlow[0])
      return true;

   if(slopeFast < -MinTrendSlope && emaFast[0] < emaSlow[0])
      return true;

   return false;
}
bool IsVolatilitySufficient()
{
   if(!UseVolatilityFilter) return true;

   double atr[1];
   if(CopyBuffer(ATR_Handle_Vol, 0, 0, 1, atr) != 1)
   {
      Print("⚠️ ATR CopyBuffer failed");
      return false;
   }

   if(atr[0] >= MinVolatilityATR * _Point)
      return true;

   PrintFormat("🔕 Market too quiet. ATR = %.1f < %.1f", atr[0]/_Point, MinVolatilityATR);
   return false;
}

//+------------------------------------------------------------------+
//| News Filter using TradingEconomics API                          |
//+------------------------------------------------------------------+
bool IsNewsHighImpact()
{
   if(StringLen(TradingEconomicsApiKey) == 0)
      return false; // Skip if no API Key

   string url = StringFormat("https://api.tradingeconomics.com/calendar?country=United%%20States&importance=3&c=%s", TradingEconomicsApiKey);
   uchar inBuf[]; ArrayResize(inBuf, 0);
   uchar outBuf[]; ArrayResize(outBuf, 0);
   string headers = "Content-Type: application/json\r\n", resultHeaders;
   int res = WebRequest("GET", url, headers, 5000, inBuf, outBuf, resultHeaders);

   if(res != 200) {
      Print("News API error: ", GetLastError());
      return false;
   }

   string response = CharArrayToString(outBuf);
   if(StringFind(response, "importance") != -1 && StringFind(response, "high") != -1)
   {
      Print("❌ High-impact news detected via TradingEconomics API.");
      return true;
   }

   return false;
}
//+------------------------------------------------------------------+
//| Gemini Sentiment API integration (คืนค่าเป็นความเชื่อมั่น)      |
//+------------------------------------------------------------------+
bool GetGeminiSentiment(double &score)
{
   // ถ้าไม่มี API Key → ข้าม
   if(StringLen(GeminiSentimentApiKey)==0)
      return false;

   string url     = StringFormat("https://api.gemini.com/sentiment?key=%s", GeminiSentimentApiKey);
   uchar inBuf[];  ArrayResize(inBuf,  0);
   uchar outBuf[]; ArrayResize(outBuf, 0);
   string headers = "Content-Type: application/json\r\n", resultHdr="";

   int res = WebRequest("GET", url, headers, 5000, inBuf, outBuf, resultHdr);
   if(res!=200)
   {
      PrintFormat("❌ Gemini API error: res=%d err=%d", res, GetLastError());
      return false;
   }

   // แปลงเป็นข้อความ JSON
   string json = CharArrayToString(outBuf, 0, WHOLE_ARRAY, CP_UTF8);

   // ตัวอย่างการ parse ง่ายๆ สมมติ JSON คืน {"sentiment":"positive","confidence":0.72}
   int p1 = StringFind(json, "\"confidence\":");
   if(p1<0)
      return false;  // ไม่เจอ
   p1 += StringLen("\"confidence\":");
   int p2 = StringFind(json, "}", p1);
   if(p2<0) p2 = StringLen(json);

   score = StringToDouble(StringSubstr(json, p1, p2-p1));
   return true;
}

//+------------------------------------------------------------------+
//| Reversal Signal Checker (Exit Condition)                        |
//+------------------------------------------------------------------+
bool ShouldExitByReversalSignal()
{
   MqlRates rates[];
   if(CopyRates(_Symbol, _Period, 0, 3, rates) < 3) return false;
   ArraySetAsSeries(rates, true);

   if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
   {
      // Bearish Engulfing for BUY
      return (rates[1].close > rates[1].open && rates[0].close < rates[0].open && rates[0].open > rates[1].close && rates[0].close < rates[1].open);
   }
   else if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
   {
      // Bullish Engulfing for SELL
      return (rates[1].close < rates[1].open && rates[0].close > rates[0].open && rates[0].open < rates[1].close && rates[0].close > rates[1].open);
   }

   return false;
}

#endif // __AI_CONDITIONS_MQH__