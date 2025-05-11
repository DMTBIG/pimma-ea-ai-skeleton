//+------------------------------------------------------------------+
//|                                     AI_ProtectProfitExit_Advanced.mqh |
//+------------------------------------------------------------------+
#ifndef __AI_PROTECT_EXIT_ADV_MQH__
#define __AI_PROTECT_EXIT_ADV_MQH__

// === IMPORT ฟังก์ชันและตัวแปรจากภายนอก ===
#include <Trade\Trade.mqh> // เพิ่ม include CTrade
#include "Settings.mqh"
#include "AI_Module.mqh"
#include "ExitLogging.mqh"

// --- ประกาศ extern สำหรับตัวแปรที่ใช้ร่วมกับไฟล์อื่น ---
extern int    ATR_Handle;            // ต้องประกาศจากไฟล์หลัก
extern CTrade trade;                 // << เพิ่ม: บอกว่า trade มาจากไฟล์หลัก
//extern bool   UseNnForSlTp;          // << เพิ่ม: บอกว่า UseNnForSlTp มาจากไฟล์หลัก
int RSI_Handle = iRSI(_Symbol, _Period, RSI_Period, PRICE_CLOSE);
extern int ATR_Handle_Vol;  // handle สำหรับ ATR volatility filter


//+------------------------------------------------------------------+
//| Trailing Fallback - SL ธรรมดา (ใช้เมื่อไม่มี AI)             |
//+------------------------------------------------------------------+
void ProtectProfit_TrailingSimple()
{
   if(!PositionSelect(_Symbol)) return;

   ulong ticket = PositionGetInteger(POSITION_TICKET);
   int type = (int)PositionGetInteger(POSITION_TYPE);
   double price = (type == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double sl = PositionGetDouble(POSITION_SL);
   double tp = PositionGetDouble(POSITION_TP);
   double step = SL_TrailingStep * _Point;

   if(type == POSITION_TYPE_BUY)
   {
      double trailSL = NormalizeDouble(price - step, _Digits);
      if(sl == 0.0 || trailSL > sl)
         trade.PositionModify(ticket, trailSL, tp);
   }
   else
   {
      double trailSL = NormalizeDouble(price + step, _Digits);
      if(sl == 0.0 || trailSL < sl)
         trade.PositionModify(ticket, trailSL, tp);
   }
}

//+------------------------------------------------------------------+
//| ProtectProfit_AI - ใช้ AI Score ช่วยประเมินการกันกำไร         |
//+------------------------------------------------------------------+
void ProtectProfit_AI(double score)
{
   if(!PositionSelect(_Symbol)) return;

   ulong ticket    = PositionGetInteger(POSITION_TICKET);
   int   type      = (int)PositionGetInteger(POSITION_TYPE);
   double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   double curSL     = PositionGetDouble(POSITION_SL);
   double curTP     = PositionGetDouble(POSITION_TP);

   double curPrice  = (type == POSITION_TYPE_BUY)
                      ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                      : SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   double profitPoints = (type == POSITION_TYPE_BUY) ? (curPrice - openPrice) : (openPrice - curPrice);
   double profitPips = profitPoints / _Point;

   double trailDistance = 0;
   if(score >= 0.85)
      trailDistance = 5.0;
   else if(score >= 0.7)
      trailDistance = 10.0;
   else if(score >= 0.55)
      trailDistance = 15.0;
   else
      trailDistance = 20.0;

   if(profitPips >= trailDistance + 5.0)
   {
      double newSL = (type == POSITION_TYPE_BUY)
                     ? NormalizeDouble(curPrice - trailDistance * _Point, _Digits)
                     : NormalizeDouble(curPrice + trailDistance * _Point, _Digits);

      bool modify = false;
      if(type == POSITION_TYPE_BUY && newSL > curSL && newSL > openPrice)
         modify = true;
      else if(type == POSITION_TYPE_SELL && (curSL == 0 || newSL < curSL) && newSL < openPrice)
         modify = true;

      if(modify)
      {
         if(!trade.PositionModify(ticket, newSL, curTP))
            PrintFormat("ProtectProfit_AI Error: Modify SL failed for ticket %d. Error: %d", ticket, GetLastError());
         else
            PrintFormat("ProtectProfit_AI: SL moved for ticket %d to %.5f (score=%.2f)", ticket, newSL, score);
      }
   }
}

//+------------------------------------------------------------------+
//| AdjustTP_DynamicZone - ใช้โซนแบบ ATR + FixZone                |
//+------------------------------------------------------------------+
void AdjustTP_DynamicZone()
{
   if(!PositionSelect(_Symbol)) return;
   double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   int type = (int)PositionGetInteger(POSITION_TYPE);
   double curSL = PositionGetDouble(POSITION_SL);
   double curTP = PositionGetDouble(POSITION_TP);
   double atrArray[1];
   if(CopyBuffer(ATR_Handle, 0, 0, 1, atrArray) != 1) return;
   double atr = atrArray[0];

   double fixedZone = 300 * _Point;
   double dynamicZone = atr * 3.0;
   double newTP = 0.0;

   if(type == POSITION_TYPE_BUY)
      newTP = NormalizeDouble(openPrice + MathMin(dynamicZone, fixedZone), _Digits);
   else
      newTP = NormalizeDouble(openPrice - MathMin(dynamicZone, fixedZone), _Digits);

   if((curTP == 0) || (type == POSITION_TYPE_BUY && newTP > curTP) || (type == POSITION_TYPE_SELL && newTP < curTP))
   {
      ulong ticket = PositionGetInteger(POSITION_TICKET);
      if(trade.PositionModify(ticket, curSL, newTP))
         PrintFormat("🏃‍♂️ AdjustTP_DynamicZone: TP updated to %.5f", newTP);
   }
}


//+------------------------------------------------------------------+
//| AdjustSL_DynamicZone - AI Score ปรับ SL แบบอัจฉริยะ          |
//+------------------------------------------------------------------+
void AdjustSL_DynamicZone(ulong ticket)
{
   // 1. ตรวจสอบว่ามีโพซิชั่น และ ticket ตรงกัน
   if(!PositionSelect(_Symbol) || PositionGetInteger(POSITION_TICKET) != ticket)
      return;

   // 2. ดึงค่า ATR ปัจจุบันจาก handle (สมมติว่า ATR_Handle สร้างไว้ใน OnInit)
   double atrArr[1];
   if(CopyBuffer(ATR_Handle, 0, 0, 1, atrArr) != 1)
   {
      Print("❌ AdjustSL: Failed to read ATR");
      return;
   }
   double atr = atrArr[0];
   if(atr <= 0) return;

   // 3. คำนวณระยะ SL (points) = ATR*multiplier + buffer pip
   double dist = atr * ATR_SL_Multiplier + SL_Buffer_Pips * _Point;

   // 4. อ่านราคาเปิด และชนิดโพซิชั่น
   double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   int    ptype     = (int)PositionGetInteger(POSITION_TYPE);
   double newSL     = (ptype == POSITION_TYPE_BUY)
                      ? openPrice - dist
                      : openPrice + dist;

   // 5. อ่าน TP ปัจจุบัน
   double curTP = PositionGetDouble(POSITION_TP);

   // 6. เช็คว่า SL ใหม่ “ดีกว่า” หรือยังไม่เคยตั้ง
   double curSL = PositionGetDouble(POSITION_SL);
   bool   modOK = false;
   if(curSL == 0.0)
      modOK = true;
   else if(ptype == POSITION_TYPE_BUY && newSL > curSL)
      modOK = true;
   else if(ptype == POSITION_TYPE_SELL && newSL < curSL)
      modOK = true;

   // 7. ถ้าได้ ให้ส่งคำสั่งแก้ SL
   if(modOK)
   {
      if(trade.PositionModify(ticket, newSL, curTP))
         PrintFormat("🚀 AdjustSL: set SL=%.5f (dist=%.1f pips)", newSL, dist/_Point);
      else
         PrintFormat("❌ AdjustSL failed: Error %d", GetLastError());
   }
}
//+------------------------------------------------------------------+
//| AdjustSL_BreakEven – ขยับ SL ไปที่ Breakeven + Offset           |
//+------------------------------------------------------------------+
void AdjustSL_BreakEven(double score=0.0)
{
   if(!EnableBreakEven || !PositionSelect(_Symbol)) return;

   ulong  ticket   = PositionGetInteger(POSITION_TICKET);
   int    type     = (int)PositionGetInteger(POSITION_TYPE);
   double entry    = PositionGetDouble(POSITION_PRICE_OPEN);
   double curPrice = (type==POSITION_TYPE_BUY)
                      ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                      : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double distPips = MathAbs(curPrice - entry) / _Point;

   // ใช้ชื่อ input ให้ตรงกับที่ประกาศไว้ใน Settings.mqh
   if(distPips < BreakEvenMinProfitPips) 
      return;

   double newSL = (type==POSITION_TYPE_BUY)
                  ? entry + BreakEvenOffsetPips * _Point
                  : entry - BreakEvenOffsetPips * _Point;

   double oldSL = PositionGetDouble(POSITION_SL);
   if(oldSL==0.0 ||
      (type==POSITION_TYPE_BUY  && newSL>oldSL) ||
      (type==POSITION_TYPE_SELL && newSL<oldSL))
   {
      if(trade.PositionModify(ticket,
                              NormalizeDouble(newSL,_Digits),
                              PositionGetDouble(POSITION_TP)))
         PrintFormat("🔐 AdjustSL_BreakEven: moved SL to %.5f (pips=%.1f)", newSL, distPips);
   }
}

//+------------------------------------------------------------------+
//| PartialExitOrSLByAI - ตัดกำไรบางส่วนหาก Score ตก และวาง SL     |
//+------------------------------------------------------------------+
void PartialExitOrSLByAI(double score)
{
   if(!PositionSelect(_Symbol)) return;

   ulong ticket = PositionGetInteger(POSITION_TICKET);
   double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   int    type      = (int)PositionGetInteger(POSITION_TYPE);
   double curPrice  = (type == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double profit    = PositionGetDouble(POSITION_PROFIT);
   double lots      = PositionGetDouble(POSITION_VOLUME);

   if(score < 0.3 && profit > 0 && lots >= 0.02)
   {
      double partialLots = NormalizeDouble(lots / 2.0, 2);
      if(partialLots >= SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN))
      {
         if(type == POSITION_TYPE_BUY)
            trade.Sell(partialLots, _Symbol, curPrice, 0, 0, "PartialExit_AI");
         else
            trade.Buy(partialLots, _Symbol, curPrice, 0, 0, "PartialExit_AI");
         PrintFormat("⚠️ PartialExitOrSLByAI: Partial exit triggered due to low score (%.2f)", score);
      }
   }
   else if(score < 0.5)
   {
      double tighterSL = (type == POSITION_TYPE_BUY) ? curPrice - 10*_Point : curPrice + 10*_Point;
      double curTP = PositionGetDouble(POSITION_TP);
      if(trade.PositionModify(ticket, tighterSL, curTP))
         PrintFormat("⚠️ PartialExitOrSLByAI: SL tightened due to low score (%.2f)", score);
   }
}

//+------------------------------------------------------------------+
//| CheckPartialExitByDynamicZone - Exit by RSI and Hold Time       |
//+------------------------------------------------------------------+
void CheckPartialExitByDynamicZone(ulong ticket)
{
   if(!PositionSelectByTicket(ticket)) return;

   datetime entryTime = (datetime)PositionGetInteger(POSITION_TIME);
   double profit = PositionGetDouble(POSITION_PROFIT);
   double lots = PositionGetDouble(POSITION_VOLUME);
   int type = (int)PositionGetInteger(POSITION_TYPE);

   // ⏳ Time filter: ถือเกิน X นาที (default 15)
   int heldMinutes = (int)((TimeCurrent() - entryTime) / 60);
   if(heldMinutes < ExitSafeHoldingMinutes || profit <= 0 || lots < 0.02)
      return;

   // 📉 RSI filter: เริ่มกลับตัวจาก overbought/oversold
   double rsi[2];
   if(CopyBuffer(RSI_Handle, 0, 0, 2, rsi) != 2) return;

   bool exitSignal = false;
   if(type == POSITION_TYPE_BUY && rsi[0] < rsi[1] && rsi[0] > RSI_Overbought - 5)
      exitSignal = true;
   if(type == POSITION_TYPE_SELL && rsi[0] > rsi[1] && rsi[0] < RSI_Oversold + 5)
      exitSignal = true;

   if(exitSignal)
   {
      double partialLots = NormalizeDouble(lots / 2.0, 2);
      if(partialLots >= SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN))
      {
         double price = (type == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         string comment = "PartialExit_DynamicZone";

         if(type == POSITION_TYPE_BUY)
            trade.Sell(partialLots, _Symbol, price, 0, 0, comment);
         else
            trade.Buy(partialLots, _Symbol, price, 0, 0, comment);

         PrintFormat("📤 DynamicZonePartialExit: %s %.2f lots at %.2f",
                     comment, partialLots, price);
      }
   }
}
//+------------------------------------------------------------------+
//| Cluster SL Protection - ถอยจาก High/Low ล่าสุด                 |
//+------------------------------------------------------------------+
void ProtectStopByCluster()
{
   if(!EnableClusterProtection || !PositionSelect(_Symbol)) return;

   ulong ticket = PositionGetInteger(POSITION_TICKET);
   int type = (int)PositionGetInteger(POSITION_TYPE);
   double sl = PositionGetDouble(POSITION_SL);
   double tp = PositionGetDouble(POSITION_TP);
   double newSL = 0.0;

   MqlRates recent[];
   if(CopyRates(_Symbol, _Period, 0, 5, recent) < 5) return;
   ArraySetAsSeries(recent, true);

   if(type == POSITION_TYPE_BUY)
   {
      double swingLow = recent[1].low;
      newSL = NormalizeDouble(swingLow - SL_ClusterOffset * _Point, _Digits);
      if(sl == 0.0 || newSL > sl)
         trade.PositionModify(ticket, newSL, tp);
   }
   else
   {
      double swingHigh = recent[1].high;
      newSL = NormalizeDouble(swingHigh + SL_ClusterOffset * _Point, _Digits);
      if(sl == 0.0 || newSL < sl)
         trade.PositionModify(ticket, newSL, tp);
   }
}



//+------------------------------------------------------------------+
//| ตรวจสอบว่าเวลาที่ถือออเดอร์ผ่านขั้นต่ำที่กำหนดหรือยัง         |
//+------------------------------------------------------------------+
bool IsSafeHoldingTimePassed()
{
   if(!PositionSelect(_Symbol)) return false;

   datetime entryTime = (datetime)PositionGetInteger(POSITION_TIME);
   datetime now = TimeCurrent();
   int elapsedMinutes = (int)((now - entryTime) / 60);

   if(elapsedMinutes >= ExitSafeHoldingMinutes)
      return true;

   PrintFormat("⏳ Holding too short: %d min < %d min", elapsedMinutes, ExitSafeHoldingMinutes);
   return false;
}
//+------------------------------------------------------------------+
//| ProtectProfit_Static – ระบบ SL/TP แบบเดิม (Static Trailing)     |
//+------------------------------------------------------------------+
// ใน Settings.mqh ตั้งเป็น StaticBreakEvenThreshold / StaticBreakEvenOffset
void ProtectProfit_Static()
{
   if(!EnableProtectProfitStatic || !PositionSelect(_Symbol))
      return;

   ulong  ticket    = PositionGetInteger(POSITION_TICKET);
   int    type      = (int)PositionGetInteger(POSITION_TYPE);
   double entry     = PositionGetDouble(POSITION_PRICE_OPEN);
   double curPrice  = (type==POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                                                : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double profitPips= MathAbs(curPrice - entry)/_Point;
   double oldSL     = PositionGetDouble(POSITION_SL);

   // 1) Break-Even
   if(profitPips >= StaticBreakEvenThreshold)
   {
      double newSL = (type==POSITION_TYPE_BUY)
                     ? entry + StaticBreakEvenOffset*_Point
                     : entry - StaticBreakEvenOffset*_Point;
      if(oldSL==0.0 || (type==POSITION_TYPE_BUY && newSL>oldSL)
                   || (type==POSITION_TYPE_SELL && newSL<oldSL))
      {
         trade.PositionModify(ticket, NormalizeDouble(newSL,_Digits),
                                            PositionGetDouble(POSITION_TP));
         PrintFormat("🔐 Static BreakEven → SL=%.5f (pips=%.1f)", newSL, profitPips);
         return; // อย่าให้ Trailing มาทับ
      }
   }

   // 2) Trailing Stop (ตามที่ตั้ง SL_TrailingStep)
   if(profitPips >= SL_TrailingStep/_Point)
   {
      int steps = (int)MathFloor((profitPips - StaticBreakEvenOffset)/ (SL_TrailingStep/_Point));
      if(steps>0)
      {
         double newSL = (type==POSITION_TYPE_BUY)
                        ? entry + steps*SL_TrailingStep
                        : entry - steps*SL_TrailingStep;
         if((type==POSITION_TYPE_BUY && newSL>oldSL)
          ||(type==POSITION_TYPE_SELL && newSL<oldSL))
            trade.PositionModify(ticket,newSL,PositionGetDouble(POSITION_TP));
      }
   }
}



#endif // __AI_PROTECT_EXIT_ADV_MQH__