"""
继电器控制服务（Jetson.GPIO）
- 默认使用 BOARD 编号，控制引脚为 18
- 提供 on/off/toggle/cleanup 接口
- 若当前环境无 Jetson.GPIO，则方法为安全空操作，便于在开发机运行 GUI
"""
from typing import Optional

try:
    import Jetson.GPIO as _GPIO  # type: ignore
except Exception:
    _GPIO = None


class RelayControl:
    """简洁的继电器控制封装。"""

    def __init__(self, pin: int = 18, mode: str = "BOARD") -> None:
        self.pin: int = pin
        self.mode: str = mode.upper()
        self._enabled: bool = False
        self._state_on: bool = False
        self._gpio = _GPIO

        if self._gpio is None:
            # Jetson.GPIO 不可用时，不抛异常，保持 GUI 可运行
            return

        if self.mode not in {"BOARD", "BCM"}:
            raise ValueError("mode must be 'BOARD' or 'BCM'")

        self._gpio.setwarnings(False)
        self._gpio.setmode(self._gpio.BOARD if self.mode == "BOARD" else self._gpio.BCM)
        # 初始为关（低电平）
        self._gpio.setup(self.pin, self._gpio.OUT, initial=self._gpio.LOW)
        self._enabled = True

    def on(self) -> None:
        """打开继电器（输出高电平，按题示例）。"""
        if not self._enabled:
            return
        self._gpio.output(self.pin, self._gpio.HIGH)
        self._state_on = True

    def off(self) -> None:
        """关闭继电器（输出低电平）。"""
        if not self._enabled:
            return
        self._gpio.output(self.pin, self._gpio.LOW)
        self._state_on = False

    def toggle(self) -> None:
        """切换继电器状态。"""
        if self._state_on:
            self.off()
        else:
            self.on()

    def cleanup(self) -> None:
        """释放 GPIO 资源。"""
        if not self._enabled:
            return
        try:
            # 清理所有已使用的通道，最安全
            self._gpio.cleanup()
        finally:
            self._enabled = False
            self._state_on = False

    def is_on(self) -> bool:
        """返回当前记录的状态（仅在本进程内有效）。"""
        return self._state_on

    def available(self) -> bool:
        """返回底层 Jetson.GPIO 是否可用。"""
        return self._enabled

    def __del__(self) -> None:
        # 避免进程退出时未清理
        try:
            self.cleanup()
        except Exception:
            pass
