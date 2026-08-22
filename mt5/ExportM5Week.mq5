// Exporta candles M5 para CSV (time,open,high,low,close,tick_volume,real_volume).
// Rode no gráfico do WIN$ (ou vencimento). Depois copie o arquivo de
// MQL5/Files para datasets/mt5_m5_week.csv no repositório trader-api.
#property script_show_inputs
#property copyright "trader-api"
#property version   "1.0"

input datetime InpFrom = D'2026.08.10 00:00:00';
input datetime InpTo   = D'2026.08.21 23:59:00';
input string   InpFile = "mt5_m5_week.csv";

void OnStart()
  {
   MqlRates rates[];
   int copied = CopyRates(_Symbol, PERIOD_M5, InpFrom, InpTo, rates);
   if(copied <= 0)
     {
      Print("CopyRates falhou: ", GetLastError());
      return;
     }

   int handle = FileOpen(InpFile, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(handle == INVALID_HANDLE)
     {
      Print("FileOpen falhou: ", GetLastError());
      return;
     }

   FileWrite(handle, "time", "open", "high", "low", "close", "tick_volume", "real_volume");
   for(int i = 0; i < copied; i++)
     {
      FileWrite(handle,
                TimeToString(rates[i].time, TIME_DATE | TIME_SECONDS),
                DoubleToString(rates[i].open, _Digits),
                DoubleToString(rates[i].high, _Digits),
                DoubleToString(rates[i].low, _Digits),
                DoubleToString(rates[i].close, _Digits),
                IntegerToString(rates[i].tick_volume),
                IntegerToString((long)rates[i].real_volume));
     }
   FileClose(handle);
   Print("Exportou ", copied, " candles M5 de ", _Symbol, " para MQL5/Files/", InpFile);
  }
