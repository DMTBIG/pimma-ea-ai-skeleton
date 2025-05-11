//+------------------------------------------------------------------+
//|                      AI_StopTradingLogic.mqh                     |
//|     รวมทุกฟังก์ชันสำหรับตรวจสอบการหยุดเทรด (Stop Trading)     |
//+------------------------------------------------------------------+
#ifndef __AI_STOPTRADINGLOGIC_MQH__ // เพิ่ม #ifndef/#define/#endif เพื่อป้องกัน Include ซ้ำ
#define __AI_STOPTRADINGLOGIC_MQH__

#include "AI_Conditions.mqh"
#include "Settings.mqh"

double lastEntryScore = 0.0;
int ConsecutiveLossCount = 0;
double CumulativePnL = 0.0;
double AIScoreHistory[10];
int AIScoreIndex = 0;

// === Trading Frequency + Smart Delay ===
datetime lastTradeTime = 0;
int MinTradeIntervalSeconds = 300; // รออย่างน้อย 5 นาที
int ConsecutiveLowAIScore = 0;
int MaxLowAIScoreAllowed = 5;

// เรียกใน OnTick หรือหลังเทรดแต่ละครั้ง
void UpdatePnLStatus(double lastTradePnL)
{
   CumulativePnL += lastTradePnL;
   if(lastTradePnL < 0)
      ConsecutiveLossCount++;
   else
      ConsecutiveLossCount = 0;
}

//+------------------------------------------------------------------+
//| Frequency Check                                                  |
//+------------------------------------------------------------------+
bool IsEntryCooldownOver()
{
   return (TimeCurrent() - lastTradeTime >= MinTradeIntervalSeconds);
}

void RecordAIScore(double score)
{
   if(score < 0.4)
      ConsecutiveLowAIScore++;
   else
      ConsecutiveLowAIScore = 0;
}

bool IsDrawdownExceeded()
{
   double maxDD = 10.0; // Max drawdown %
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double dd = 100.0 * (1.0 - equity / balance);
   if(dd >= maxDD)
   {
      PrintFormat("🛑 StopTrading: Drawdown %.2f%% ≥ %.2f%%", dd, maxDD);
      return true;
   }
   return false;
}


bool IsLowAIScoreStreak()
{
   return (ConsecutiveLowAIScore >= MaxLowAIScoreAllowed);
   /*
   int count = 0;
   int limit = MathMin(AIScoreIndex, 10);
   for(int i = 0; i < limit; i++)
   {
      if(AIScoreHistory[i] < 0.4)
         count++;
   }
   return (count >= AILowScoreThreshold);
   */
}

bool IsTimeBlocked()
{
   // --- เพิ่มการตรวจสอบ Enable/Disable ---
   // ตรวจสอบค่าจาก Input ที่เพิ่มใน Settings.mqh ก่อน
   if (!EnableTimeBlock) // ถ้า Input EnableTimeBlock ถูกตั้งค่าเป็น false (ปิดใช้งาน)
   {
      return false;      // ให้คืนค่า false (ไม่ Block) ทันที โดยไม่ต้องเช็คเวลา
   }
   // --- ถ้า EnableTimeBlock เป็น true จะทำงานต่อตามปกติ ---

   datetime now = TimeLocal();
   string currentTimeStr = TimeToString(now, TIME_MINUTES);

   // ตรวจสอบเงื่อนไขเวลาเหมือนเดิม
   if((BlockStartTime < BlockEndTime && currentTimeStr >= BlockStartTime && currentTimeStr <= BlockEndTime) ||
      (BlockStartTime > BlockEndTime && (currentTimeStr >= BlockStartTime || currentTimeStr <= BlockEndTime)))
      return true; // ถ้าเข้าเงื่อนไขเวลา ให้ Block

   // ถ้าไม่เข้าเงื่อนไขเวลา
   return false; // ไม่ Block
}

//+------------------------------------------------------------------+
//| คำนวณจำนวนแท่งเทียนที่ถือ Position นี้ไว้แล้ว                |
//+------------------------------------------------------------------+
int BarsSinceEntry()
{
   if(!PositionSelect(_Symbol)) return 0;

   datetime entryTime = (datetime)PositionGetInteger(POSITION_TIME);
   int barsAgo = iBarShift(_Symbol, _Period, entryTime);
   return (barsAgo >= 0) ? barsAgo : 0;
}

//+------------------------------------------------------------------+
//| ดึงผลกำไร/ขาดทุนของดีลล่าสุดจาก history                        |
//+------------------------------------------------------------------+
double GetLastClosedTradeProfit()
{
   int deals = HistoryDealsTotal();
   if(deals < 1) 
      return(0.0);

   ulong ticket = HistoryDealGetTicket(deals - 1);
   if(HistoryDealSelect(ticket))
      // เรียกพร้อมระบุ ID ของ property
      return( HistoryDealGetDouble(ticket, DEAL_PROFIT) );

   return(0.0);
}


//+------------------------------------------------------------------+
//| ตรวจสอบเงื่อนไขหยุดเทรดทั้งหมด                                 |
//+------------------------------------------------------------------+
bool ShouldStopTrading()
{
   // 1) ปิดการเทรดโดยรวม
   if(!EnableTrading)
   {
      Print("🛑 StopTrading: Trading disabled");
      return(true);
   }
   // 2) Cooldown …
   if(TimeCurrent() - lastTradeTime < MinTradeIntervalSeconds)
   {
      PrintFormat("🛑 StopTrading: Cooldown Active (%d sec)", MinTradeIntervalSeconds);
      return(true);
   }
   
   // 3) ขาดทุนติดต่อกัน ...
   static int lossStreak = 0;
   double lastPnL = GetLastClosedTradeProfit();
   if(lastPnL < 0.0) lossStreak++; else lossStreak = 0;
   if(lossStreak >= MaxConsecutiveLosses)
   {
      PrintFormat("🛑 StopTrading: MaxConsecutiveLosses=%d reached", lossStreak);
      return(true);
   }
   // 4) ขาดทุนสะสม ...
   static double cumLoss = 0.0;
   if(lastPnL < 0.0) cumLoss += lastPnL;
   if(cumLoss <= MaxCumulativeLoss)
   {
      PrintFormat("🛑 StopTrading: MaxCumulativeLoss=%.2f reached (cumLoss=%.2f)", MaxCumulativeLoss, cumLoss);
      return(true);
   }
   // 5) AI Score ต่ำติดต่อกัน ...
   static int aiLowStreak = 0;
   if(lastEntryScore <= AI_EntryThresholdLow) aiLowStreak++; else aiLowStreak = 0;
   if(aiLowStreak >= AILowScoreThreshold)
   {
      PrintFormat("🛑 StopTrading: Low AI score streak=%d ≥ %d", aiLowStreak, AILowScoreThreshold);
      return(true);
   }
   // 6) Drawdown Pips ...
   double ddPips = (AccountInfoDouble(ACCOUNT_BALANCE) - AccountInfoDouble(ACCOUNT_EQUITY)) / _Point;
   if(ddPips >= MaxDrawdownPips)
   {
      PrintFormat("🛑 StopTrading: DrawdownPips=%.1f ≥ %d", ddPips, MaxDrawdownPips);
      return(true);
   }
   // 7) Drawdown % ...
   double ddPercent = 100.0 * (AccountInfoDouble(ACCOUNT_BALANCE) - AccountInfoDouble(ACCOUNT_EQUITY))
                      / AccountInfoDouble(ACCOUNT_BALANCE);
   if(ddPercent >= MaxAllowedDrawdownPercent)
   {
      PrintFormat("🛑 StopTrading: Drawdown%%=%.2f%% ≥ %.2f%%", ddPercent, MaxAllowedDrawdownPercent);
      return(true);
   }
   // 8) ถือแท่งเทียนนานเกิน ...
   if(PositionSelect(_Symbol))
   {
      int barsHeld = BarsSinceEntry();
      if(barsHeld >= MaxBarsHolding)
      {
         PrintFormat("🛑 StopTrading: Held %d bars ≥ %d", barsHeld, MaxBarsHolding);
         return(true);
      }
   }
   // 9) Time‑Block / News‑Block ...
   if(EnableTimeBlock && IsTimeBlocked())
   {
      Print("🛑 StopTrading: Time-based block active");
      return(true);
   }
   if(CheckNewsHighImpact())
   {
      Print("🛑 StopTrading: High impact news detected");
      return(true);
   }

   // ผ่านทั้งหมด → เทรดได้
   return(false);
}


#endif // __AI_STOPTRADINGLOGIC_MQH__
