#!/usr/bin/env python3
"""
简易继电器图形界面（基于 Tkinter）
- 使用现有的 RelayControl 封装
- 提供“打开/关闭/切换”按钮和状态显示
- 在未安装 Jetson.GPIO 的环境中，按钮将被禁用，但界面可正常运行
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from typing import Optional

from relay_control import RelayControl


class RelayGUI:
    """继电器控制的简易 GUI。"""

    def __init__(self, root: tk.Tk, pin: int = 18, mode: str = "BOARD") -> None:
        self.root: tk.Tk = root
        self.root.title("继电器控制")
        self.root.geometry("360x200")

        self.relay: RelayControl = RelayControl(pin=pin, mode=mode)

        self.status_var: tk.StringVar = tk.StringVar()
        self.btn_on: Optional[tk.Button] = None
        self.btn_off: Optional[tk.Button] = None
        self.btn_toggle: Optional[tk.Button] = None

        self._build_ui(pin=pin, mode=mode)
        self._refresh_controls()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- UI ----------
    def _build_ui(self, pin: int, mode: str) -> None:
        container = tk.Frame(self.root, padx=12, pady=12)
        container.pack(fill=tk.BOTH, expand=True)

        status_label = tk.Label(
            container,
            textvariable=self.status_var,
            font=("Arial", 14, "bold"),
        )
        status_label.pack(pady=(0, 8))

        btn_frame = tk.Frame(container)
        btn_frame.pack(fill=tk.X)

        self.btn_on = tk.Button(
            btn_frame,
            text=" 打开继电器",
            command=self.turn_on,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11),
            cursor="hand2",
            relief=tk.RAISED,
            borderwidth=2,
        )
        self.btn_on.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))

        self.btn_off = tk.Button(
            btn_frame,
            text=" 关闭继电器",
            command=self.turn_off,
            bg="#F44336",
            fg="white",
            font=("Arial", 11),
            cursor="hand2",
            relief=tk.RAISED,
            borderwidth=2,
        )
        self.btn_off.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))

        # 可选的“切换”按钮
        self.btn_toggle = tk.Button(
            container,
            text=" 切换开关",
            command=self.toggle,
            bg="#607D8B",
            fg="white",
            font=("Arial", 11),
            cursor="hand2",
            relief=tk.RAISED,
            borderwidth=2,
        )
        self.btn_toggle.pack(fill=tk.X, pady=(8, 0))

        hint = tk.Label(
            container,
            text=f"默认模式: {mode}，引脚: {pin}",
            fg="#666666",
            font=("Arial", 10),
        )
        hint.pack(pady=(8, 0))

    # ---------- Actions ----------
    def turn_on(self) -> None:
        if not self.relay.available():
            messagebox.showwarning("提示", "当前环境未安装 Jetson.GPIO，无法控制继电器")
            return
        try:
            self.relay.on()
        finally:
            self._refresh_controls()

    def turn_off(self) -> None:
        if not self.relay.available():
            messagebox.showwarning("提示", "当前环境未安装 Jetson.GPIO，无法控制继电器")
            return
        try:
            self.relay.off()
        finally:
            self._refresh_controls()

    def toggle(self) -> None:
        if not self.relay.available():
            messagebox.showwarning("提示", "当前环境未安装 Jetson.GPIO，无法控制继电器")
            return
        try:
            self.relay.toggle()
        finally:
            self._refresh_controls()

    def _refresh_controls(self) -> None:
        available = self.relay.available()
        if not available:
            self.status_var.set("继电器不可用（Jetson.GPIO 未安装）")
        else:
            self.status_var.set("继电器状态：开" if self.relay.is_on() else "继电器状态：关")

        state = tk.NORMAL if available else tk.DISABLED
        if self.btn_on is not None:
            self.btn_on.config(state=state)
        if self.btn_off is not None:
            self.btn_off.config(state=state)
        if self.btn_toggle is not None:
            self.btn_toggle.config(state=state)

    # ---------- Lifecycle ----------
    def on_close(self) -> None:
        try:
            self.relay.cleanup()
        except Exception:
            pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = RelayGUI(root, pin=18, mode="BOARD")
    root.mainloop()


if __name__ == "__main__":
    main()
