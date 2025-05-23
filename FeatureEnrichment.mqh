//+------------------------------------------------------------------+
//|                                          FeatureEnrichment.mqh   |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com"

#include "Settings.mqh"             // ต้องมี PythonServiceURL

//+------------------------------------------------------------------+
//| Google Trends via /trends?kw=…                                    |
//+------------------------------------------------------------------+
double GetGoogleTrends(const string keyword)
{
   string url = PythonServiceURL + "/trends?kw=" + keyword;
   string hdr = "Content-Type: application/json\r\n";
   uchar  inBuf[];  ArrayResize(inBuf,0);
   uchar  outBuf[]; ArrayResize(outBuf,0);
   string respHdr="";
   int res = WebRequest("GET",url,hdr,5000,inBuf,outBuf,respHdr);
   if(res!=200) return(0.0);
   string json = CharArrayToString(outBuf,0,WHOLE_ARRAY,CP_UTF8);
   // parse {"value":NN.NN}
   int p = StringFind(json,"\"value\":");
   if(p<0) return(0.0);
   int start = p+StringLen("\"value\":");
   int e1 = StringFind(json, ",", start), e2 = StringFind(json, "}", start);
   int end = (e1>=0 && e1<e2 ? e1 : e2);
   string v = (end>=0 ? StringSubstr(json, start, end-start) : StringSubstr(json,start));
   StringTrimLeft(v); StringTrimRight(v);
   return(StringToDouble(v));
}

//+------------------------------------------------------------------+
//| Social Sentiment via /social?src=twitter|reddit                  |
//+------------------------------------------------------------------+
double GetSocialSentiment(const string source)
{
   string url = PythonServiceURL + "/social?src=" + source;
   string hdr = "Content-Type: application/json\r\n";
   uchar  inBuf[];  ArrayResize(inBuf,0);
   uchar  outBuf[]; ArrayResize(outBuf,0);
   string respHdr="";
   int res = WebRequest("GET",url,hdr,5000,inBuf,outBuf,respHdr);
   if(res!=200) return(0.0);
   string json = CharArrayToString(outBuf,0,WHOLE_ARRAY,CP_UTF8);
   int p = StringFind(json,"\"value\":");
   if(p<0) return(0.0);
   int start = p+StringLen("\"value\":");
   int e1 = StringFind(json, ",", start), e2 = StringFind(json, "}", start);
   int end = (e1>=0 && e1<e2 ? e1 : e2);
   string v = (end>=0 ? StringSubstr(json, start, end-start) : StringSubstr(json,start));
   StringTrimLeft(v); StringTrimRight(v);
   return(StringToDouble(v));
}

//+------------------------------------------------------------------+
//| Macro Indicator via /macro?name=…                                |
//+------------------------------------------------------------------+
double GetMacroIndicator(const string name)
{
   string url = PythonServiceURL + "/macro?name=" + name;
   string hdr = "Content-Type: application/json\r\n";
   uchar  inBuf[];  ArrayResize(inBuf,0);
   uchar  outBuf[]; ArrayResize(outBuf,0);
   string respHdr="";
   int res = WebRequest("GET",url,hdr,10000,inBuf,outBuf,respHdr);
   if(res!=200) return(0.0);
   string json = CharArrayToString(outBuf,0,WHOLE_ARRAY,CP_UTF8);
   int p = StringFind(json,"\"value\":");
   if(p<0) return(0.0);
   int start = p+StringLen("\"value\":");
   int e1 = StringFind(json, ",", start), e2 = StringFind(json, "}", start);
   int end = (e1>=0 && e1<e2 ? e1 : e2);
   string v = (end>=0 ? StringSubstr(json, start, end-start) : StringSubstr(json,start));
   StringTrimLeft(v); StringTrimRight(v);
   return(StringToDouble(v));
}

//+------------------------------------------------------------------+
//| Order Book Imbalance via /orderbook?symbol=…                     |
//+------------------------------------------------------------------+
double GetOrderBookImbalance(const string symbol)
{
   string url = PythonServiceURL + "/orderbook?symbol=" + symbol;
   string hdr = "Content-Type: application/json\r\n";
   uchar  inBuf[];  ArrayResize(inBuf,0);
   uchar  outBuf[]; ArrayResize(outBuf,0);
   string respHdr="";
   int res = WebRequest("GET",url,hdr,5000,inBuf,outBuf,respHdr);
   if(res!=200) return(0.0);
   return(StringToDouble(CharArrayToString(outBuf,0,WHOLE_ARRAY,CP_UTF8)));
}

//+------------------------------------------------------------------+
//| On‐chain Data via /onchain?metric=active_addresses&symbol=…     |
//+------------------------------------------------------------------+
double GetOnchainActiveAddresses(const string ticker)
{
   string url = PythonServiceURL + "/onchain?metric=active_addresses&symbol=" + ticker;
   string hdr = "Content-Type: application/json\r\n";
   uchar  inBuf[];  ArrayResize(inBuf,0);
   uchar  outBuf[]; ArrayResize(outBuf,0);
   string respHdr="";
   int res = WebRequest("GET",url,hdr,5000,inBuf,outBuf,respHdr);
   if(res!=200) return(0.0);
   return(StringToDouble(CharArrayToString(outBuf,0,WHOLE_ARRAY,CP_UTF8)));
}
