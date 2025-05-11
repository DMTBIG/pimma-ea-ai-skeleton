#!/usr/bin/env bash
MT5_PATH="$HOME/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"
wine "$MT5_PATH" /compile:"$PWD/PIMMA_EA_AI_Skeleton_Clean.mq5"
wine "$MT5_PATH" /compile:"$PWD/AI_Module.mqh"
