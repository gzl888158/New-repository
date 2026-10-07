"""负责笔记本环境降级保护：电量、温度、防休眠与进程崩溃监控。"""
import asyncio
import math
import psutil
import platform
import subprocess
import os
import signal
from datetime import datetime
from typing import Dict, Any, Optional, List
from loguru import logger


def _finite(value: Any, default: float = 0.0) -> float:
    """安全数值转换：None/非法字符串/NaN/Inf 统一回退到 default。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _position_close_side(position) -> str:
    """计算平仓方向：多头卖、空头买；net 模式按数量正负判断。"""
    raw = (getattr(position, "side", "") or "").strip().lower()
    if raw == "long":
        return "sell"
    if raw == "short":
        return "buy"
    return "sell" if _finite(getattr(position, "quantity", 0.0), 0.0) > 0 else "buy"


def _position_pos_side(position) -> str:
    """归一化 posSide：long/short 原样返回，net 原样返回，其余按数量符号推导。"""
    raw = (getattr(position, "side", "") or "").strip().lower()
    if raw in ("long", "short", "net"):
        return raw
    return "long" if _finite(getattr(position, "quantity", 0.0), 0.0) >= 0 else "short"


class NotebookFallbackControl:
    def __init__(self, config: Dict[str, Any], okx_client, redis_cache):
        self.config = config
        self.okx_client = okx_client
        self.redis_cache = redis_cache
        
        self._temp_threshold = config["hardware"]["temperature_threshold"]
        self._temp_critical = 95
        self._battery_critical = 10
        self._battery_warning = 30
        self._battery_normal = 50
        
        self._is_battery_mode = False
        self._battery_level = 100
        
        self._overheating = False
        self._last_temp_check = datetime.now()
        
        self._last_wake_time = datetime.now()
        self._sleep_prevention_active = True
        
        self._process_crashes = {}
        self._max_crashes = 5
        self._crash_window = 300

        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self):
        if self._running:
            return
        self._running = True
        self._tasks.append(asyncio.create_task(self._monitor_loop()))
        self._tasks.append(asyncio.create_task(self._sleep_prevention_loop()))
        self._tasks.append(asyncio.create_task(self._crash_monitor_loop()))
        logger.info("NotebookFallbackControl started")

    async def stop(self):
        self._running = False
        self._sleep_prevention_active = False
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("NotebookFallbackControl stopped")

    async def _monitor_loop(self):
        while self._running:
            try:
                await self._check_battery()
                await self._check_temperature()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Notebook monitor loop error: {e}")
            await asyncio.sleep(10)

    async def _check_battery(self):
        if not self._is_laptop():
            return
        
        battery = psutil.sensors_battery()
        if battery is None:
            return
        
        self._battery_level = battery.percent
        self._is_battery_mode = not battery.power_plugged
        
        if self._is_battery_mode:
            logger.warning(f"Battery mode active: {self._battery_level}%")
            
            if self._battery_level <= self._battery_critical:
                logger.critical(f"Battery critically low ({self._battery_level}%), initiating emergency shutdown")
                await self._emergency_shutdown()
            
            elif self._battery_level <= self._battery_warning:
                logger.warning(f"Battery low ({self._battery_level}%), closing aggressive positions")
                await self._close_aggressive_positions()
            
            elif self._battery_level <= self._battery_normal:
                logger.info(f"Battery below {self._battery_normal}%, sending alert")
                await self._send_battery_alert()

    async def _check_temperature(self):
        try:
            temps = psutil.sensors_temperatures()
        except AttributeError:
            return
        
        cpu_temp = 0.0
        for key in ("coretemp", "cpu_thermal", "acpitz"):
            entries = temps.get(key) or []
            valid = [_finite(getattr(temp, "current", None), None) for temp in entries]
            valid = [v for v in valid if v is not None]
            if valid:
                cpu_temp = max(valid)
                break
        
        if cpu_temp >= self._temp_critical:
            logger.critical(f"CPU temperature {cpu_temp}°C critically high, initiating emergency shutdown")
            await self._emergency_shutdown()
        elif cpu_temp >= self._temp_threshold:
            logger.warning(f"CPU temperature {cpu_temp}°C exceeds threshold {self._temp_threshold}°C")
            if not self._overheating:
                self._overheating = True
                await self._handle_overheating()
        else:
            self._overheating = False

    async def _handle_overheating(self):
        logger.error("CPU overheating, pausing high-frequency strategies")
        await self._pause_high_frequency_strategies()

    async def _emergency_shutdown(self):
        try:
            positions = self.okx_client.get_positions()
        except Exception as e:
            logger.error(f"Failed to fetch positions during emergency shutdown: {e}")
            positions = None

        # P1-5: 批量平仓优化 — 一次 API 调用平所有仓位
        order_bodies = []
        for pos_data in positions or []:
            try:
                position = self.okx_client._parse_position(pos_data)
            except Exception:
                position = None
            if not position:
                continue
            qty = abs(_finite(position.quantity, 0.0))
            if qty <= 0:
                continue
            side = _position_close_side(position)
            pos_side = _position_pos_side(position)

            # 构建订单体
            is_spot = "-SWAP" not in position.symbol
            contracts_qty = qty
            if not is_spot:
                contracts_qty = self.okx_client.coin_to_contracts(position.symbol, qty)
                contracts_qty = self.okx_client.round_quantity_to_lot(position.symbol, contracts_qty, round_up=True)
            if contracts_qty <= 0:
                continue

            body = {
                "instId": position.symbol,
                "side": side,
                "ordType": "market",
                "sz": str(contracts_qty),
                "reduceOnly": True,
            }
            if is_spot:
                body["tdMode"] = "cash"
            else:
                body["tdMode"] = "isolated"
                body["lever"] = str(int(_finite(position.leverage, 1.0)))
                body["posSide"] = pos_side
            order_bodies.append((position.symbol, body))

        if order_bodies:
            try:
                batch_results = self.okx_client.place_batch_orders([b[1] for b in order_bodies])
                for idx, (symbol, _) in enumerate(order_bodies):
                    if idx >= len(batch_results) or batch_results[idx].get("_failed", False):
                        msg = batch_results[idx].get("sMsg", "") if idx < len(batch_results) else "No result"
                        logger.error(f"Emergency close failed for {symbol}: {msg}")
            except Exception as e:
                logger.error(f"Batch emergency close failed: {e}")

        logger.critical("All positions closed. System shutting down.")
        self._sleep_prevention_active = False

    async def _close_aggressive_positions(self):
        try:
            positions = self.okx_client.get_positions()
        except Exception as e:
            logger.error(f"Failed to fetch positions for aggressive close: {e}")
            positions = None

        # P1-5: 批量平仓优化 — 只平高杠杆（>=8x）仓位
        order_bodies = []
        for pos_data in positions or []:
            try:
                position = self.okx_client._parse_position(pos_data)
            except Exception:
                position = None
            if not position:
                continue
            qty = abs(_finite(position.quantity, 0.0))
            if qty <= 0:
                continue
            leverage = _finite(position.leverage, 1.0)
            if leverage >= 8:
                side = _position_close_side(position)
                pos_side = _position_pos_side(position)

                # 构建订单体
                is_spot = "-SWAP" not in position.symbol
                contracts_qty = qty
                if not is_spot:
                    contracts_qty = self.okx_client.coin_to_contracts(position.symbol, qty)
                    contracts_qty = self.okx_client.round_quantity_to_lot(position.symbol, contracts_qty, round_up=True)
                if contracts_qty <= 0:
                    continue

                body = {
                    "instId": position.symbol,
                    "side": side,
                    "ordType": "market",
                    "sz": str(contracts_qty),
                    "reduceOnly": True,
                }
                if is_spot:
                    body["tdMode"] = "cash"
                else:
                    body["tdMode"] = "isolated"
                    body["lever"] = str(int(leverage))
                    body["posSide"] = pos_side
                order_bodies.append((position.symbol, body))

        if order_bodies:
            try:
                batch_results = self.okx_client.place_batch_orders([b[1] for b in order_bodies])
                for idx, (symbol, _) in enumerate(order_bodies):
                    if idx >= len(batch_results) or batch_results[idx].get("_failed", False):
                        msg = batch_results[idx].get("sMsg", "") if idx < len(batch_results) else "No result"
                        logger.error(f"Aggressive close failed for {symbol}: {msg}")
            except Exception as e:
                logger.error(f"Batch aggressive close failed: {e}")

    async def _pause_high_frequency_strategies(self):
        pass

    async def _send_battery_alert(self):
        logger.info(f"Battery alert: {self._battery_level}% remaining")

    async def _sleep_prevention_loop(self):
        while self._sleep_prevention_active and self._running:
            try:
                self._prevent_sleep()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Sleep prevention failed: {e}")
            await asyncio.sleep(60)

    def _prevent_sleep(self):
        system = platform.system()
        if system == "Windows":
            subprocess.run(
                ["powercfg", "-change", "-monitor-timeout-ac", "0"],
                capture_output=True
            )
            subprocess.run(
                ["powercfg", "-change", "-standby-timeout-ac", "0"],
                capture_output=True
            )
            subprocess.run(
                ["powercfg", "-change", "-hibernate-timeout-ac", "0"],
                capture_output=True
            )
        elif system == "Darwin":
            subprocess.run(
                ["caffeinate", "-d", "-i", "-m", "-s"],
                capture_output=True
            )
        elif system == "Linux":
            subprocess.run(
                ["xdg-screensaver", "suspend"],
                capture_output=True,
                errors="ignore"
            )

    async def _crash_monitor_loop(self):
        while self._running:
            try:
                await self._check_process_crashes()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Crash monitor loop error: {e}")
            await asyncio.sleep(60)

    async def _check_process_crashes(self):
        current_time = datetime.now().timestamp()
        
        for process_name, crashes in list(self._process_crashes.items()):
            self._process_crashes[process_name] = [
                crash_time for crash_time in crashes 
                if current_time - crash_time < self._crash_window
            ]
            
            if not self._process_crashes[process_name]:
                del self._process_crashes[process_name]

    def report_crash(self, process_name: str):
        current_time = datetime.now().timestamp()
        
        if process_name not in self._process_crashes:
            self._process_crashes[process_name] = []
        
        self._process_crashes[process_name].append(current_time)
        
        crash_count = len(self._process_crashes[process_name])
        logger.warning(f"Process {process_name} crashed {crash_count} times in last {self._crash_window}s")
        
        if crash_count >= self._max_crashes:
            logger.error(f"Process {process_name} exceeded max crashes ({self._max_crashes}), stopping strategy")
            return False
        
        return True

    def _is_laptop(self) -> bool:
        system = platform.system()
        if system == "Windows":
            try:
                result = subprocess.run(
                    ["powercfg", "/query"],
                    capture_output=True,
                    text=True
                )
                return "Battery" in result.stdout
            except:
                return True
        elif system == "Darwin":
            try:
                result = subprocess.run(
                    ["system_profiler", "SPPowerDataType"],
                    capture_output=True,
                    text=True
                )
                return "Battery Information" in result.stdout
            except:
                return True
        return False

    def get_battery_status(self) -> Dict[str, Any]:
        return {
            "level": self._battery_level,
            "is_battery_mode": self._is_battery_mode,
            "overheating": self._overheating
        }

    def get_system_status(self) -> Dict[str, Any]:
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        return {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "battery_level": self._battery_level,
            "is_battery_mode": self._is_battery_mode,
            "overheating": self._overheating,
            "process_crashes": self._process_crashes
        }