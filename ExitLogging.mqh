#include "Settings.mqh"

//+------------------------------------------------------------------+
//| LogExitFailureToServer                                           |
//+------------------------------------------------------------------+
void LogExitFailureToServer(ulong ticket,string reason)
{
   string url     = PythonServiceURL + "/log_execution_failure";
   string headers = "Content-Type: application/json\r\n\r\n";

   // 1) สร้าง payload เป็น string
   string payload = StringFormat(
      "{\"ticket\":%Iu,\"reason\":\"%s\"}",
       ticket, reason);

   // 2) แปลง payload → uchar[]
   uchar postData[];
   int   len = StringToCharArray(payload, postData, CP_UTF8);
   ArrayResize(postData, len);

   // 3) เตรียม buffer รับ response
   uchar  response[];
   string resp_headers;

   // 4) เรียก WebRequest ด้วย overload ถูกต้อง
   int status = WebRequest(
      "POST",      // method
      url,         // URL
      headers,     // headers (ต้องมี CRLF สองครั้งตอนจบ)
      5000,        // timeout(ms)
      postData,    // request body as uchar[]
      response,    // response body buffer
      resp_headers // response headers
   );

   // 5) แปลง response → string แล้วพิมพ์ debug
   string body = CharArrayToString(response);
   PrintFormat(
     "🔴 LogExitFailure → HTTP %d, err=%d\nResp-Headers:\n%s\nResp-Body:\n%s",
     status, GetLastError(), resp_headers, body
   );
}

//+------------------------------------------------------------------+
//| LogExitDecisionToServer                                          |
//+------------------------------------------------------------------+
void LogExitDecisionToServer(ulong ticket,double score,string method)
{
   string url     = PythonServiceURL + "/log_exit_decision";
   string headers = "Content-Type: application/json\r\n\r\n";

   // 1) สร้าง payload
   string payload = StringFormat(
      "{\"ticket\":%Iu,\"score\":%.4f,\"method\":\"%s\"}",
      ticket, score, method);

   // 2) แปลง → uchar[]
   uchar postData[];
   int   len = StringToCharArray(payload, postData, CP_UTF8);
   ArrayResize(postData, len);

   // 3) เตรียมรับ response
   uchar  response[];
   string resp_headers;

   // 4) เรียก WebRequest
   int status = WebRequest(
      "POST",
      url,
      headers,
      5000,
      postData,
      response,
      resp_headers
   );

   // 5) แปลง response → string แล้ว debug
   string body = CharArrayToString(response);
   PrintFormat(
     "🔴 LogExitDecision → HTTP %d, err=%d\nResp-Headers:\n%s\nResp-Body:\n%s",
     status, GetLastError(), resp_headers, body
   );
}
