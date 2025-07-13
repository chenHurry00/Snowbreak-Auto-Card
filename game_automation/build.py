from cx_Freeze import setup, Executable
import sys

icon_file = "Acacia.ico"

executables = [
    Executable(
        "game.py",
        icon=icon_file,  # 指定图标
        target_name="GameApp.exe"  # 输出文件名
    )
]

build_options = {
    "build_exe": {
        "include_files": [
            ("card_templates/", "card_templates/"), 
            ("player_templates/", "player_templates/"),
        ],
        "excludes": ["tkinter", "unittest"],  # 排除无用库
        "optimize": 2
    }
}

setup(
    name="GameApp",
    version="1.0",
    description="Game Automation Script",
    options=build_options,
    executables=executables
)