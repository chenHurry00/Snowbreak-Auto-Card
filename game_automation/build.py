from cx_Freeze import setup, Executable
setup(
    executables=[Executable("game.py")],
    options={
        "build_exe": {
            "include_files": [("card_templates/", "player_templates/")]  # 包含图片文件夹
        }
    }
)